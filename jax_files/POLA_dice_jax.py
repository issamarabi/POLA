# Some parts adapted from https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
# Some parts adapted from Chris Lu's MOFOS repo


# TODO: FULL CODE REVIEW AND COMMENT EVERYTHING THAT IS HAPPENING.

import numpy as np
import argparse
import datetime
from typing import Tuple, Any

import jax
from jax import jit
import optax
from functools import partial

from flax import linen as nn
import jax.numpy as jnp
from flax.training.train_state import TrainState


from flax.training import checkpoints

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


from coin_game_jax import CoinGame
from ipd_jax import IPD

def reverse_cumsum(x, axis):
    return x + jnp.sum(x, axis=axis, keepdims=True) - jnp.cumsum(x, axis=axis)

# DiCE operator
@jit
def magic_box(x):
    return jnp.exp(x - jax.lax.stop_gradient(x))

@jit
def update_gae_with_delta_backwards(gae, delta):
    gae = gae * args.gamma * args.gae_lambda + delta
    return gae, gae

@jit
def get_gae_advantages(rewards, values, next_val_history):
    deltas = rewards + args.gamma * jax.lax.stop_gradient(
        next_val_history) - jax.lax.stop_gradient(values)

    gae = jnp.zeros_like(deltas[0, :])

    deltas = jnp.flip(deltas, axis=0)
    gae, flipped_advantages = jax.lax.scan(update_gae_with_delta_backwards, gae, deltas, deltas.shape[0])
    advantages = jnp.flip(flipped_advantages, axis=0)

    return advantages

@jit
def dice_objective(self_logprobs, other_logprobs, rewards, values, end_state_v):
    # apply discount:
    cum_discount = jnp.cumprod(args.gamma * jnp.ones(rewards.shape),
                                 axis=0) / args.gamma
    discounted_rewards = rewards * cum_discount

    # stochastics nodes involved in rewards dependencies:
    dependencies = jnp.cumsum(self_logprobs + other_logprobs, axis=0)

    # logprob of all stochastic nodes:
    stochastic_nodes = self_logprobs + other_logprobs

    use_loaded_dice = False
    if use_baseline:
        use_loaded_dice = True

    if use_loaded_dice:
        next_val_history = jnp.zeros((args.rollout_len, args.batch_size))

        next_val_history = next_val_history.at[:args.rollout_len - 1, :].set(values[1:args.rollout_len, :])
        next_val_history = next_val_history.at[-1, :].set(end_state_v)

        if args.zero_vals:
            next_val_history = jnp.zeros_like(next_val_history)
            values = jnp.zeros_like(values)

        advantages = get_gae_advantages(rewards, values, next_val_history)

        discounted_advantages = advantages * cum_discount

        deps_up_to_t = (jnp.cumsum(stochastic_nodes, axis=0))

        deps_less_than_t = deps_up_to_t - stochastic_nodes  # take out the dependency in the given time step

        # Look at Loaded DiCE and GAE papers to see where this formulation comes from
        loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(
            deps_less_than_t)) * discounted_advantages).sum(axis=0).mean()

        dice_obj = loaded_dice_rewards

    else:
        # dice objective:
        # REMEMBER that in this jax code the axis 0 is the rollout_len (number of time steps in the environment)
        # and axis 1 is the batch.
        dice_obj = jnp.mean(
            jnp.sum(magic_box(dependencies) * discounted_rewards, axis=0))


    return -dice_obj  # want to minimize -objective


@jit
def dice_objective_plus_value_loss(self_logprobs, other_logprobs, rewards, values, end_state_v):
    # Essentially a wrapper function for the objective to put all the control flow in one spot
    # The reasoning behind this function here is that the reward_loss has a stop_gradient
    # on all of the nodes related to the value function
    # and the value function has no nodes related to the policy
    # Then we can actually take the respective grads like the way I have things set up now
    # And I should be able to update both policy and value functions

    reward_loss = dice_objective(self_logprobs, other_logprobs, rewards, values, end_state_v)

    if use_baseline:
        val_loss = value_loss(rewards, values, end_state_v)
        return reward_loss + val_loss
    else:
        return reward_loss


@jit
def value_loss(rewards, values, final_state_vals):

    final_state_vals = jax.lax.stop_gradient(final_state_vals)

    discounts = jnp.cumprod(args.gamma * jnp.ones(rewards.shape),
                                 axis=0) / args.gamma

    gamma_t_r_ts = rewards * discounts

    # sum of discounted rewards (discounted to the first time step); first entry has all the future discounted rewards,
    # second entry has all the rewards from the second step onwards, but discounted to the first time step!
    # Thus, dividing by the cumulative discount brings the discounted rewards to the appropriate time step
    # e.g. after dividing by discounts, you now have the rewards from time step 2 onwards discounted
    # only up to time step 2
    G_ts = reverse_cumsum(gamma_t_r_ts, axis=0)
    R_ts = G_ts / discounts

    final_val_discounted_to_curr = (args.gamma * jnp.flip(discounts, axis=0)) * final_state_vals

    # You DO need a detach on these. Because it's the target - it should be detached. It's a target value.
    # Essentially a Monte Carlo style type return for R_t, except for the final state we also use the estimated final state value.
    # This becomes our target for the value function loss. So it's kind of a mix of Monte Carlo and bootstrap, but anyway you need the final value
    # because otherwise your value calculations will be inconsistent
    values_loss = (R_ts + final_val_discounted_to_curr - values) ** 2

    values_loss = values_loss.sum(axis=0).mean()

    return values_loss


@jit
def act_w_iter_over_obs(stuff, env_batch_obs):
    key, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v = stuff
    key, subkey = jax.random.split(key)
    act_args, act_aux = generate_action(subkey, env_batch_obs, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    _, env_batch_obs, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v = act_args
    stuff = (key, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    return stuff, act_aux


@jax.jit
def generate_action(key: jax.random.PRNGKey, env_states: jnp.ndarray, policy_trainstate: TrainState, policy_params: jnp.ndarray, 
        value_trainstate: TrainState, value_params: jnp.ndarray, hidden_policy: jnp.ndarray, hidden_value: jnp.ndarray):
    """
    Generate an action for an agent given its policy and value function.
    
    Parameters:
        key (jax.random.PRNGKey): Random key for generating actions.
        env_states (jnp.ndarray): Current environment states.
        policy_trainstate (TrainState): TrainState object for the policy.
        policy_params (jnp.ndarray): Parameters for the policy.
        value_trainstate (TrainState): TrainState object fozr the value function.
        value_params (jnp.ndarray): Parameters for the value function.
        hidden_policy (jnp.ndarray): Hidden state for the policy.
        hidden_value (jnp.ndarray): Hidden state for the value function.
        
    Returns:
        tuple: Updated states and auxiliary information.
    """
    # Compute logits for the agent's actions based on the current environment state
    hidden_policy, logits = policy_trainstate.apply_fn(policy_params, env_states, hidden_policy)
    
    # Compute softmax probabilities of the actions
    action_probs = jax.nn.softmax(logits)
    
    # Sample an action based on the logits
    action_dist = tfd.Categorical(logits=logits)
    key, subkey = jax.random.split(key)
    actions = action_dist.sample(seed=subkey)
    
    # Compute log probabilities of the sampled actions
    log_probs_actions = action_dist.log_prob(actions)
    
    # Compute value estimates for the current state if using a baseline
    if use_baseline:
        hidden_value, values = value_trainstate.apply_fn(value_params, env_states, hidden_value)
        value_estimates = values.squeeze(-1)
    else:
        hidden_value, value_estimates = None, None

    return (key, env_states, policy_trainstate, policy_params, value_trainstate, value_params, hidden_policy, hidden_value), (actions, log_probs_actions, value_estimates, hidden_policy, hidden_value, action_probs, logits)


class RNN(nn.Module):
    """
    Simple RNN model with optional dense layers before the GRU cell.

    Attributes:
    - num_outputs: Number of output units.
    - num_hidden_units: Number of hidden units.
    - layers_before_gru: Number of dense layers before the GRU cell.
    """
    num_outputs: int
    num_hidden_units: int
    layers_before_gru: int

    def setup(self):
        # Define dense layers before GRU
        self.linears = [nn.Dense(features=self.num_hidden_units) for _ in range(self.layers_before_gru)]
        self.GRUCell = nn.GRUCell(features=self.num_hidden_units)  # Provide the required 'features' argument
        self.linear_end = nn.Dense(features=self.num_outputs)

    def __call__(self, x, carry):
        # Pass through dense layers
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:  # Only apply ReLU if it's not the last layer
                x = nn.relu(x)

        # Pass through GRU cell
        carry, x = self.GRUCell(carry, x)
        outputs = self.linear_end(x)
        return carry, outputs



@jit
def get_policies_for_states(key, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, obs_hist):

    h_p = jnp.zeros((args.batch_size, args.hidden_size))
    h_v = None
    if use_baseline:
        h_v = jnp.zeros((args.batch_size, args.hidden_size))

    key, subkey = jax.random.split(key)

    act_args = (subkey, th_p_trainstate, th_p_trainstate_params,
                th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    # Note that I am scanning using xs = obs_hist. Then the scan should work through the
    # array of obs.
    obs_hist_for_scan = jnp.stack(obs_hist[:args.rollout_len], axis=0) # skips final obs (but includes init/start state/obs)

    act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, args.rollout_len)
    # act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, obs_hist_for_scan.shape[0])

    a_list, lp_list, v_list, h_p_list, h_v_list, cat_act_probs_list, logits_list = aux_lists


    return cat_act_probs_list


# Same as above except just also return values
@jit
def get_policies_and_values_for_states(key, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, obs_hist):

    h_p = jnp.zeros((args.batch_size, args.hidden_size))
    h_v = None
    if use_baseline:
        h_v = jnp.zeros((args.batch_size, args.hidden_size))

    key, subkey = jax.random.split(key)

    act_args = (subkey, th_p_trainstate, th_p_trainstate_params,
                th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    # Note that I am scanning using xs = obs_hist. Then the scan should work through the
    # array of obs.
    obs_hist_for_scan = jnp.stack(obs_hist[:args.rollout_len], axis=0) # skips final obs (but includes init/start state/obs)

    act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, args.rollout_len)
    # act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, obs_hist_for_scan.shape[0])

    a_list, lp_list, v_list, h_p_list, h_v_list, cat_act_probs_list, logits_list = aux_lists


    return cat_act_probs_list, v_list

# Same as before again, but now returning the h_p and h_v. Only used for OM right now
def get_policies_and_h_for_states(key, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, obs_hist):

    h_p = jnp.zeros((args.batch_size, args.hidden_size))
    h_v = None
    if use_baseline:
        h_v = jnp.zeros((args.batch_size, args.hidden_size))

    key, subkey = jax.random.split(key)

    act_args = (subkey, th_p_trainstate, th_p_trainstate_params,
                th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    # Note that I am scanning using xs = obs_hist. Then the scan should work through the
    # array of obs.
    obs_hist_for_scan = jnp.stack(obs_hist[:args.rollout_len], axis=0) # skips final obs (but includes init/start state/obs)

    act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, args.rollout_len)
    # act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, obs_hist_for_scan.shape[0])

    a_list, lp_list, v_list, h_p_list, h_v_list, cat_act_probs_list, logits_list = aux_lists


    return cat_act_probs_list, h_p_list, h_v_list

# Do for only a single batch; only used for testing/inspection
@jit
def get_policies_for_states_onebatch(key, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, obs_hist):

    h_p = jnp.zeros((1, args.hidden_size))
    h_v = None
    if use_baseline:
        h_v = jnp.zeros((1, args.hidden_size))

    key, subkey = jax.random.split(key)

    act_args = (subkey, th_p_trainstate, th_p_trainstate_params,
                th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    # Note that I am scanning using xs = obs_hist. Then the scan should work through the
    # array of obs.
    obs_hist_for_scan = jnp.stack(obs_hist[:len(obs_hist)], axis=0)

    # act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, args.rollout_len)
    act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, obs_hist_for_scan.shape[0])

    a_list, lp_list, v_list, h_p_list, h_v_list, cat_act_probs_list, logits_list = aux_lists


    return cat_act_probs_list


@jit
def env_step(env_and_agent_states: Tuple, unused: Any) -> Tuple:
    """
    Perform one step in the environment for both agents.
    
    Parameters:
        env_and_agent_states (Tuple): Tuple containing all the necessary information for the environment and agents.
        unused (Any): Unused parameter, required for compatibility with jax.lax.scan.
        
    Returns:
        Tuple: Updated states and auxiliary information.
    """
    # Unpack states
    (key, env_state, obs_agent1, obs_agent2, 
     train_state_agent1, train_state_params1, val_state_agent1, val_state_params1,
     train_state_agent2, train_state_params2, val_state_agent2, val_state_params2,
     hidden_policy1, hidden_value1, hidden_policy2, hidden_value2) = env_and_agent_states
     
    # Generate new subkeys
    key, subkey_agent1, subkey_agent2, subkey_env = jax.random.split(key, 4)
    
    # Agent 1 action
    new_states_agent1, aux_info_agent1 = generate_action(
        subkey_agent1, obs_agent1, 
        train_state_agent1, train_state_params1, 
        val_state_agent1, val_state_params1, 
        hidden_policy1, hidden_value1
    )
    a1, lp1, v1, h_p1, h_v1, cat_act_probs1, logits1 = aux_info_agent1
    
    # Agent 2 action
    new_states_agent2, aux_info_agent2 = generate_action(
        subkey_agent2, obs_agent2, 
        train_state_agent2, train_state_params2, 
        val_state_agent2, val_state_params2, 
        hidden_policy2, hidden_value2
    )
    a2, lp2, v2, h_p2, h_v2, cat_act_probs2, logits2 = aux_info_agent2
    
    # Environment step
    subkeys_env = jax.random.split(subkey_env, args.batch_size)
    env_state, new_obs, rewards, aux_env_info = vec_env_step(env_state, aux_info_agent1[0], aux_info_agent2[0], subkeys_env)
    
    # Update observations
    obs1 = new_obs
    obs2 = new_obs
    
    # Prepare return values
    updated_states = (
        key, env_state, obs1, obs2,
        train_state_agent1, train_state_params1, val_state_agent1, val_state_params1,
        train_state_agent2, train_state_params2, val_state_agent2, val_state_params2,
        hidden_policy1, hidden_value1, hidden_policy2, hidden_value2
    )
    
    aux1 = (cat_act_probs1, obs1, lp1, lp2, v1, rewards[0], a1, a2)
    aux2 = (cat_act_probs2, obs2, lp2, lp1, v2, rewards[1], a2, a1)

    aux_info = (aux1, aux2, aux_env_info)
    
    return updated_states, aux_info


@partial(jit, static_argnums=(9))
def do_env_rollout(key, trainstate_th1, trainstate_th1_params, trainstate_val1,
             trainstate_val1_params,
             trainstate_th2, trainstate_th2_params, trainstate_val2,
             trainstate_val2_params, agent_for_state_history):
    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    env_state, obsv = vec_env_reset(env_subkeys)

    obs1 = obsv
    obs2 = obsv

    h_p1, h_p2, h_v1, h_v2 = get_init_hidden_states()

    unfinished_state_history = []
    if agent_for_state_history == 2:
        unfinished_state_history.append(obs2)
    else:
        assert agent_for_state_history == 1
        unfinished_state_history.append(obs1)

    stuff = (key, env_state, obs1, obs2,
             trainstate_th1, trainstate_th1_params, trainstate_val1,
             trainstate_val1_params,
             trainstate_th2, trainstate_th2_params, trainstate_val2,
             trainstate_val2_params,
             h_p1, h_v1, h_p2, h_v2)

    stuff, aux = jax.lax.scan(env_step, stuff, None, args.rollout_len)

    # unfinished_state_history contains just a single starting obs/state at this point
    # THen the additional observations during the rollout are added on later
    # This seems a bit weird but I guess it works. Not exactly sure why I structured it this way
    return stuff, aux, unfinished_state_history

# Do rollouts and calculate objectives for the inner agent (the other_agent)
@partial(jit, static_argnums=(11))
def in_lookahead(key, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
                 trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,
                 old_trainstate_th, old_trainstate_val,
                 other_agent=2):

    stuff, aux, unfinished_inner_agent_state_history = do_env_rollout(key, trainstate_th1, trainstate_th1_params, trainstate_val1,
             trainstate_val1_params,
             trainstate_th2, trainstate_th2_params, trainstate_val2,
             trainstate_val2_params, agent_for_state_history=other_agent)
    aux1, aux2, aux_info = aux

    inner_agent_state_history = unfinished_inner_agent_state_history

    key, env_state, obs1, obs2, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,\
    trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params, h_p1, h_v1, h_p2, h_v2 = stuff

    key, subkey1, subkey2 = jax.random.split(key, 3)

    # TODO remove redundancies in the code
    if other_agent == 2:
        cat_act_probs2_list, obs2_list, lp2_list, lp1_list, v2_list, r2_list, a2_list, a1_list = aux2

        inner_agent_state_history.extend(obs2_list)

        # act just to get the final state values
        stuff2, aux2 = generate_action(subkey2, obs2, trainstate_th2, trainstate_th2_params,
                           trainstate_val2, trainstate_val2_params, h_p2, h_v2)
        a2, lp2, v2, h_p2, h_v2, cat_act_probs2, logits2 = aux2

        end_state_v2 = v2

        inner_agent_objective = dice_objective_plus_value_loss(self_logprobs=lp2_list,
                                               other_logprobs=lp1_list,
                                               rewards=r2_list,
                                               values=v2_list,
                                               end_state_v=end_state_v2)

        # print(f"Inner Agent (Agent 2) episode return avg {r2_list.sum(axis=0).mean()}")


    else:
        assert other_agent == 1
        cat_act_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1
        inner_agent_state_history.extend(obs1_list)

        stuff1, aux1 = generate_action(subkey1, obs1, trainstate_th1, trainstate_th1_params,
                     trainstate_val1, trainstate_val1_params, h_p1, h_v1)
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1, logits1 = aux1

        end_state_v1 = v1

        inner_agent_objective = dice_objective_plus_value_loss(self_logprobs=lp1_list,
                                               other_logprobs=lp2_list,
                                               rewards=r1_list,
                                               values=v1_list,
                                               end_state_v=end_state_v1)

        # print(f"Inner Agent (Agent 1) episode return avg {r1_list.sum(axis=0).mean()}")

    key, sk1, sk2 = jax.random.split(key, 3)


    if other_agent == 2:
        inner_agent_pol_probs = get_policies_for_states(sk1,
                                                        trainstate_th2,
                                                        trainstate_th2_params,
                                                        trainstate_val2,
                                                        trainstate_val2_params,
                                                        inner_agent_state_history)
        # We don't need gradient on the old one, so we can just use the trainstate.params
    else:
        inner_agent_pol_probs = get_policies_for_states(sk1,
                                                        trainstate_th1,
                                                        trainstate_th1_params,
                                                        trainstate_val1,
                                                        trainstate_val1_params,
                                                        inner_agent_state_history)
    # NOTE the distinction between .params and _params... really confusing, and I
    # definitely should not ever do confusing notation like that again - use something that is more obviously distinct
    # But the key point is that the _params is a separate variable
    # Which I used for the grad calculations
    # But the old_trainstate (the reference for KL div) should not have a grad on it
    # so I just use the . to access its trainstate params here
    inner_agent_pol_probs_old = get_policies_for_states(sk2,
                                                        old_trainstate_th,
                                                        old_trainstate_th.params,
                                                        old_trainstate_val,
                                                        old_trainstate_val.params,
                                                        inner_agent_state_history)

    # Note that Kl Div right now (not the old kl div) is based on the state history of this episode
    # Passed through the policies of the current agent policy params and the old params
    # So what this means is that on each inner step, you get a fresh batch of data
    # For the KL Div calculation too
    # This I think should be more stable than before
    # This means you aren't limited to KL Div only on the 4000 or whatever batch
    # you got from the very beginning
    # And so you should get coverage on a wider range of the state space
    # in the same way that your updates are based on new rollouts too
    # If we do repeat train, then the repeat train KL Div should be based on the
    # initial trajectory
    # and then I have to figure out how to save the initial trajectory and reuse it in Jax.

    if args.rev_kl:
        kl_div = rev_kl_div_jax(inner_agent_pol_probs, inner_agent_pol_probs_old)
    else:
        kl_div = kl_div_jax(inner_agent_pol_probs, inner_agent_pol_probs_old)
    # print(f"KL Div: {kl_div}")

    return inner_agent_objective + args.inner_beta * kl_div  # we want to min kl div


# We have D_KL(p || q) = E [p (log p - log q)]
# The forward formulation is p = target, q = curr (target * (log target - log curr))
# The reverse formulation is p = curr, q = target (curr * (log curr - log target))
@jit
def kl_div_jax(curr, target):
    kl_div = (target * (jnp.log(target) - jnp.log(curr))).sum(axis=-1).mean()
    return kl_div

@jit
def rev_kl_div_jax(curr, target):
    kl_div = (curr * (jnp.log(curr) - jnp.log(target))).sum(axis=-1).mean()
    return kl_div

# This is a single step of update (inner loop)
@jit
def inner_step_get_grad_otheragent2(stuff, unused):
    key, trainstate_th1_, trainstate_th1_params, trainstate_val1_, trainstate_val1_params, \
    trainstate_th2_, trainstate_th2_params, trainstate_val2_, trainstate_val2_params, old_trainstate_th, old_trainstate_val = stuff
    key, subkey = jax.random.split(key)

    other_agent_obj_grad_fn = jax.grad(in_lookahead, argnums=[6, 8])

    grad_th, grad_v = other_agent_obj_grad_fn(subkey,
                                              trainstate_th1_,
                                              trainstate_th1_params,
                                              trainstate_val1_,
                                              trainstate_val1_params,
                                              trainstate_th2_,
                                              trainstate_th2_params,
                                              trainstate_val2_,
                                              trainstate_val2_params,
                                              old_trainstate_th,
                                              old_trainstate_val,
                                              other_agent=2)

    # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE
    trainstate_th2_ = trainstate_th2_.apply_gradients(grads=grad_th)

    # In old code I didn't update value function on inner loop but also I only used 1 inner step in most experiments
    if use_baseline:
        trainstate_val2_ = trainstate_val2_.apply_gradients(grads=grad_v)

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    # Note the dot here (on agent 2) because we want to return the updated params
    stuff = (key, trainstate_th1_, trainstate_th1_params, trainstate_val1_, trainstate_val1_params,
             trainstate_th2_, trainstate_th2_.params, trainstate_val2_, trainstate_val2_.params,
             old_trainstate_th, old_trainstate_val)
    aux = None

    return stuff, aux

# This is a single step of update (inner loop)
@jit
def inner_step_get_grad_otheragent1(stuff, unused):
    key, trainstate_th1_, trainstate_th1_params, trainstate_val1_, trainstate_val1_params, \
    trainstate_th2_, trainstate_th2_params, trainstate_val2_, trainstate_val2_params, old_trainstate_th, old_trainstate_val  = stuff
    key, subkey = jax.random.split(key)

    other_agent_obj_grad_fn = jax.grad(in_lookahead,
                                       argnums=[2, 4])

    grad_th, grad_v = other_agent_obj_grad_fn(subkey,
                                              trainstate_th1_,
                                              trainstate_th1_params,
                                              trainstate_val1_,
                                              trainstate_val1_params,
                                              trainstate_th2_,
                                              trainstate_th2_params,
                                              trainstate_val2_,
                                              trainstate_val2_params,
                                              old_trainstate_th, old_trainstate_val,
                                              other_agent=1)

    # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE

    trainstate_th1_ = trainstate_th1_.apply_gradients(grads=grad_th)

    # In old code I didn't update value function on inner loop but also I only used 1 inner step in most experiments
    if use_baseline:
        trainstate_val1_ = trainstate_val1_.apply_gradients(grads=grad_v)

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    # Note the dot here (on agent 1) because we want to return the updated params
    stuff = (key, trainstate_th1_, trainstate_th1_.params, trainstate_val1_, trainstate_val1_.params,
             trainstate_th2_, trainstate_th2_params, trainstate_val2_, trainstate_val2_params,
             old_trainstate_th, old_trainstate_val)
    aux = None

    return stuff, aux

# This does all the inner steps + updates (for one outer step)
@jit
def inner_steps_plus_update_otheragent2(key, trainstate_th1, trainstate_th1_params,
                            trainstate_val1, trainstate_val1_params,
                            trainstate_th2, trainstate_th2_params,
                            trainstate_val2, trainstate_val2_params,
                            other_old_trainstate_th, other_old_trainstate_val):


    trainstate_th2_prime = TrainState.create(apply_fn=trainstate_th2.apply_fn,
                                        params=trainstate_th2_params,
                                        tx=optax.sgd(
                                            learning_rate=args.lr_in))
    trainstate_val2_prime = TrainState.create(apply_fn=trainstate_val2.apply_fn,
                                         params=trainstate_val2_params,
                                         tx=optax.sgd(
                                             learning_rate=args.lr_v))


    key, subkey = jax.random.split(key)



    # preserving the params we want to diff through on the outer loop (th1)
    stuff = (subkey, trainstate_th1, trainstate_th1_params,
             trainstate_val1, trainstate_val1_params,
             trainstate_th2_prime, trainstate_th2_prime.params,
             trainstate_val2_prime, trainstate_val2_prime.params, other_old_trainstate_th,
             other_old_trainstate_val)

    stuff, aux = inner_step_get_grad_otheragent2(stuff, None)

    _, _, _, _, _, trainstate_th2_prime, _, trainstate_val2_prime, _, _, _ = stuff

    key, subkey = jax.random.split(key)

    if args.inner_steps > 1:
        stuff = (subkey, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
                 trainstate_th2_prime, trainstate_th2_prime.params,
                 trainstate_val2_prime, trainstate_val2_prime.params,
                 other_old_trainstate_th, other_old_trainstate_val)
        # The way this scan loop works: each time the trainstate_th2 (and val) is updated in each loop iter
        # And its .params are returned after each loop iteration as well
        # so each new loop takes the gradient using the .params of the previous iteration
        # while updating the trainstate itself to have the new .params
        # In this way the loop iterates updating the inner agent
        stuff, aux = jax.lax.scan(inner_step_get_grad_otheragent2, stuff,
                                  None, args.inner_steps - 1)
        _, _, _, _, _, trainstate_th2_prime, _, trainstate_val2_prime, _, _, _ = stuff

    if use_baseline:
        return trainstate_th2_prime, trainstate_val2_prime
    else:
        return trainstate_th2_prime, None


# This does all the inner steps + updates (for one outer step)
@jit
def inner_steps_plus_update_otheragent1(key, trainstate_th1, trainstate_th1_params,
                            trainstate_val1, trainstate_val1_params,
                            trainstate_th2, trainstate_th2_params,
                            trainstate_val2, trainstate_val2_params,
                            other_old_trainstate_th, other_old_trainstate_val):

    trainstate_th1_prime = TrainState.create(apply_fn=trainstate_th1.apply_fn,
                                        params=trainstate_th1_params,
                                        tx=optax.sgd(
                                            learning_rate=args.lr_in))
    trainstate_val1_prime = TrainState.create(apply_fn=trainstate_val1.apply_fn,
                                         params=trainstate_val1_params,
                                         tx=optax.sgd(
                                             learning_rate=args.lr_v))

    key, subkey = jax.random.split(key)


    # preserving the params we want to diff through on the outer loop (th2)
    stuff = (subkey, trainstate_th1_prime, trainstate_th1_prime.params,
             trainstate_val1_prime, trainstate_val1_prime.params,
             trainstate_th2, trainstate_th2_params,
             trainstate_val2, trainstate_val2_params, other_old_trainstate_th,
             other_old_trainstate_val)

    stuff, aux = inner_step_get_grad_otheragent1(stuff, None)

    _, trainstate_th1_prime, _, trainstate_val1_prime, _, _, _, _, _, _, _ = stuff

    key, subkey = jax.random.split(key)

    if args.inner_steps > 1:
        stuff = (subkey, trainstate_th1_prime, trainstate_th1_prime.params,
                 trainstate_val1_prime, trainstate_val1_prime.params,
                 trainstate_th2, trainstate_th2_params,
                 trainstate_val2, trainstate_val2_params,
                 other_old_trainstate_th, other_old_trainstate_val)
        stuff, aux = jax.lax.scan(inner_step_get_grad_otheragent1, stuff,
                                  None, args.inner_steps - 1)
        _, trainstate_th1_prime, _, trainstate_val1_prime, _, _, _, _, _, _, _ = stuff

    if use_baseline:
        return trainstate_th1_prime, trainstate_val1_prime
    else:
        return trainstate_th1_prime, None


# Do rollouts and calculate objectives for the outer agent (the self_agent)
@partial(jit, static_argnums=(11))
def out_lookahead(key, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
                  trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,
                  old_trainstate_th, old_trainstate_val, self_agent=1):

    stuff, aux, unfinished_state_history_for_kl_div = do_env_rollout(key, trainstate_th1,
                                                           trainstate_th1_params,
                                                           trainstate_val1,
                                                           trainstate_val1_params,
                                                           trainstate_th2,
                                                           trainstate_th2_params,
                                                           trainstate_val2,
                                                           trainstate_val2_params,
                                                           agent_for_state_history=self_agent)

    aux1, aux2, aux_info = aux
    state_history_for_kl_div = unfinished_state_history_for_kl_div

    # This is equivalent to just redeclaring stuff like trainstate_th1_params = trainstate_th1_params
    # Because it is unmodified from the env rollouts.
    # So this redeclaration is unnecessary (but also shouldn't have any effect)
    key, env_state, obs1, obs2, \
    trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,\
    trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,\
    h_p1, h_v1, h_p2, h_v2 = stuff

    if self_agent == 1:
        cat_act_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1

        # cat_act_probs_self.extend(cat_act_probs1_list)
        state_history_for_kl_div.extend(obs1_list)

        key, subkey = jax.random.split(key)
        # act just to get the final state values

        stuff1, aux1 = generate_action(subkey, obs1, trainstate_th1, trainstate_th1_params,
                     trainstate_val1, trainstate_val1_params, h_p1, h_v1)
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1, logits1 = aux1

        end_state_v = v1
        objective = dice_objective_plus_value_loss(self_logprobs=lp1_list,
                                   other_logprobs=lp2_list,
                                   rewards=r1_list, values=v1_list,
                                   end_state_v=end_state_v)
        # print(f"Agent 1 episode return avg {r1_list.sum(axis=0).mean()}")
    else:
        assert self_agent == 2
        cat_act_probs2_list, obs2_list, lp2_list, lp1_list, v2_list, r2_list, a2_list, a1_list = aux2

        state_history_for_kl_div.extend(obs2_list)

        key, subkey = jax.random.split(key)
        # act just to get the final state values
        stuff2, aux2 = generate_action(subkey, obs2, trainstate_th2, trainstate_th2_params,
                     trainstate_val2, trainstate_val2_params, h_p2, h_v2)
        a2, lp2, v2, h_p2, h_v2, cat_act_probs2, logits2 = aux2

        end_state_v = v2
        objective = dice_objective_plus_value_loss(self_logprobs=lp2_list,
                                   other_logprobs=lp1_list,
                                   rewards=r2_list, values=v2_list,
                                   end_state_v=end_state_v)
        # print(f"Agent 2 episode return avg {r2_list.sum(axis=0).mean()}")

    key, sk1, sk2 = jax.random.split(key, 3)


    if self_agent == 1:
        self_pol_probs = get_policies_for_states(sk1, trainstate_th1,
                                                 trainstate_th1_params,
                                                 trainstate_val1,
                                                 trainstate_val1_params,
                                                 state_history_for_kl_div)
    else:
        self_pol_probs = get_policies_for_states(sk1,
                                                 trainstate_th2,
                                                 trainstate_th2_params,
                                                 trainstate_val2,
                                                 trainstate_val2_params,
                                                 state_history_for_kl_div)

    self_pol_probs_ref = get_policies_for_states(sk2,
                                                        old_trainstate_th,
                                                        old_trainstate_th.params,
                                                        old_trainstate_val,
                                                        old_trainstate_val.params,
                                                        state_history_for_kl_div)

    if args.rev_kl:
        kl_div = rev_kl_div_jax(self_pol_probs, self_pol_probs_ref)
    else:
        kl_div = kl_div_jax(self_pol_probs, self_pol_probs_ref)

    # return grad
    return objective + args.outer_beta * kl_div, state_history_for_kl_div



@jit
def one_outer_step_objective_selfagent1(key, trainstate_th1_copy, trainstate_th1_copy_params, trainstate_val1_copy, trainstate_val1_copy_params,
                             trainstate_th2_copy, trainstate_th2_copy_params, trainstate_val2_copy, trainstate_val2_copy_params,
                             trainstate_th_ref, trainstate_val_ref):
    self_agent = 1
    other_agent = 2
    key, subkey = jax.random.split(key)
    trainstate_th2_after_inner_steps, trainstate_val2_after_inner_steps = \
        inner_steps_plus_update_otheragent2(subkey,
                                trainstate_th1_copy, trainstate_th1_copy_params,
                                trainstate_val1_copy,
                                trainstate_val1_copy_params,
                                trainstate_th2_copy, trainstate_th2_copy_params,
                                trainstate_val2_copy,
                                trainstate_val2_copy_params,
                                trainstate_th2_copy, trainstate_val2_copy
                                )
    # It's a bit weird to have trainstate_th2_copy show up twice in the above,
    # but I believe it ends up all working out fine because inner_steps_plus_update
    # makes a copy of trainstate_th2_copy before taking updates on it
    # so that won't affect the second trainstate_th2_copy which is used as the reference
    # point for KL div.

    if use_baseline:
        objective, state_hist_from_rollout = out_lookahead(key, trainstate_th1_copy,
                                  trainstate_th1_copy_params,
                                  trainstate_val1_copy,
                                  trainstate_val1_copy_params,
                                  trainstate_th2_after_inner_steps,
                                  trainstate_th2_after_inner_steps.params,
                                  trainstate_val2_after_inner_steps,
                                  trainstate_val2_after_inner_steps.params,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent)
    else:
        objective, state_hist_from_rollout = out_lookahead(key, trainstate_th1_copy,
                                  trainstate_th1_copy_params,
                                  None, None,
                                  trainstate_th2_after_inner_steps,
                                  trainstate_th2_after_inner_steps.params,
                                  None, None,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent)

    return objective, state_hist_from_rollout


@jit
def one_outer_step_objective_selfagent2(key, trainstate_th1_copy, trainstate_th1_copy_params, trainstate_val1_copy, trainstate_val1_copy_params,
                             trainstate_th2_copy, trainstate_th2_copy_params, trainstate_val2_copy, trainstate_val2_copy_params,
                             trainstate_th_ref, trainstate_val_ref):
    self_agent = 2
    other_agent = 1
    key, subkey = jax.random.split(key)
    trainstate_th1_after_inner_steps, trainstate_val1_after_inner_steps = \
        inner_steps_plus_update_otheragent1(subkey,
                                trainstate_th1_copy, trainstate_th1_copy_params,
                                trainstate_val1_copy,
                                trainstate_val1_copy_params,
                                trainstate_th2_copy, trainstate_th2_copy_params,
                                trainstate_val2_copy,
                                trainstate_val2_copy_params,
                                trainstate_th1_copy, trainstate_val1_copy)


    if use_baseline:
        objective, state_hist_from_rollout = out_lookahead(key, trainstate_th1_after_inner_steps,
                                  trainstate_th1_after_inner_steps.params,
                                  trainstate_val1_after_inner_steps,
                                  trainstate_val1_after_inner_steps.params,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy_params,
                                  trainstate_val2_copy,
                                  trainstate_val2_copy_params,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent)
    else:
        objective, state_hist_from_rollout = out_lookahead(key, trainstate_th1_after_inner_steps,
                                  trainstate_th1_after_inner_steps.params,
                                  None, None,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy_params,
                                  None, None,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent)

    return objective, state_hist_from_rollout

@jit
def one_outer_step_update_selfagent1(stuff, unused):
    key, trainstate_th1_copy, trainstate_val1_copy, trainstate_th2_copy, trainstate_val2_copy, \
    trainstate_th_ref, trainstate_val_ref = stuff

    key, subkey = jax.random.split(key)

    obj_grad_fn = jax.grad(one_outer_step_objective_selfagent1, argnums=[2, 4], has_aux=True)

    (grad_th, grad_v), state_hist_from_rollout = obj_grad_fn(subkey,
                                  trainstate_th1_copy,
                                  trainstate_th1_copy.params,
                                  trainstate_val1_copy,
                                  trainstate_val1_copy.params,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy.params,
                                  trainstate_val2_copy,
                                  trainstate_val2_copy.params,
                                  trainstate_th_ref, trainstate_val_ref)

    trainstate_th1_copy = trainstate_th1_copy.apply_gradients(grads=grad_th)

    if use_baseline:
        trainstate_val1_copy = trainstate_val1_copy.apply_gradients(grads=grad_v)

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    stuff = (key, trainstate_th1_copy,  trainstate_val1_copy, trainstate_th2_copy,  trainstate_val2_copy,
    trainstate_th_ref, trainstate_val_ref)
    aux = state_hist_from_rollout

    return stuff, aux

@jit
def one_outer_step_update_selfagent2(stuff, unused):
    key, trainstate_th1_copy, trainstate_val1_copy, \
    trainstate_th2_copy, trainstate_val2_copy,\
    trainstate_th_ref, trainstate_val_ref = stuff


    key, subkey = jax.random.split(key)

    obj_grad_fn = jax.grad(one_outer_step_objective_selfagent2, argnums=[6, 8], has_aux=True)

    (grad_th, grad_v), state_hist_from_rollout = obj_grad_fn(subkey,
                                  trainstate_th1_copy,
                                  trainstate_th1_copy.params,
                                  trainstate_val1_copy,
                                  trainstate_val1_copy.params,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy.params,
                                  trainstate_val2_copy,
                                  trainstate_val2_copy.params,
                                  trainstate_th_ref, trainstate_val_ref)

    trainstate_th2_copy = trainstate_th2_copy.apply_gradients(grads=grad_th)

    if use_baseline:
        trainstate_val2_copy = trainstate_val2_copy.apply_gradients(grads=grad_v)

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    stuff = (
    key, trainstate_th1_copy, trainstate_val1_copy,
    trainstate_th2_copy, trainstate_val2_copy,
    trainstate_th_ref, trainstate_val_ref)
    aux = state_hist_from_rollout

    return stuff, aux

@jit
def eval_vs_alld_selfagent1(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v = stuff

    key, subkey = jax.random.split(key)

    stuff, aux = generate_action(subkey, obsv, trainstate_th, trainstate_th.params,
                     trainstate_val,trainstate_val.params, h_p, h_v)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = True
    opp_is_red_agent = False

    if args.env == "ipd":
        # Always defect
        a_opp = jnp.zeros_like(a)
    elif args.env == "coin":
        a_opp = env.get_moves_shortest_path_to_coin(env_state,
                                                    opp_is_red_agent)

    a1 = a
    a2 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v)
    aux = (score1, score2)

    return stuff, aux


@jit
def eval_vs_alld_selfagent2(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v = stuff

    key, subkey = jax.random.split(key)

    stuff, aux = generate_action(subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
                     trainstate_val.params, h_p, h_v)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = False
    opp_is_red_agent = True

    if args.env == "ipd":
        # Always defect
        a_opp = jnp.zeros_like(a)
    elif args.env == "coin":
        a_opp = env.get_moves_shortest_path_to_coin(env_state,
                                                    opp_is_red_agent)

    a2 = a
    a1 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v)
    aux = (score1, score2)

    return stuff, aux

@jit
def eval_vs_allc_selfagent1(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v = stuff

    key, subkey = jax.random.split(key)

    stuff, aux = generate_action(subkey, obsv, trainstate_th, trainstate_th.params,
                     trainstate_val, trainstate_val.params, h_p, h_v)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = True
    opp_is_red_agent = False

    if args.env == "ipd":
        # Always cooperate
        a_opp = jnp.ones_like(a)
    elif args.env == "coin":
        a_opp = env.get_coop_action(env_state, opp_is_red_agent)

    a1 = a
    a2 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v)
    aux = (score1, score2)

    return stuff, aux


@jit
def eval_vs_allc_selfagent2(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v = stuff

    key, subkey = jax.random.split(key)

    stuff, aux = generate_action(subkey, obsv, trainstate_th, trainstate_th.params,
                     trainstate_val, trainstate_val.params, h_p, h_v)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = False
    opp_is_red_agent = True

    if args.env == "ipd":
    # Always cooperate
        a_opp = jnp.ones_like(a)
    elif args.env == "coin":
        a_opp = env.get_coop_action(env_state, opp_is_red_agent)

    a2 = a
    a1 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v)
    aux = (score1, score2)

    return stuff, aux


@jit
def eval_vs_tft_selfagent1(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, prev_a, prev_agent_coin_collected_same_col, r1, r2 = stuff

    key, subkey = jax.random.split(key)

    stuff, aux = generate_action(subkey, obsv, trainstate_th, trainstate_th.params,
                     trainstate_val, trainstate_val.params, h_p, h_v)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    if args.env == "ipd":
        # Copy last move of agent; assumes prev_a = all coop
        a_opp = prev_a
        prev_agent_coin_collected_same_col = None
    elif args.env == "coin":
        r_opp = r2
        # Agent here means me, the agent we are testing
        prev_agent_coin_collected_same_col = jnp.where(r_opp < 0, 0, prev_agent_coin_collected_same_col)
        prev_agent_coin_collected_same_col = jnp.where(r_opp > 0, 1, prev_agent_coin_collected_same_col)

        a_opp_defect = env.get_moves_shortest_path_to_coin(env_state, False)
        a_opp_coop = env.get_coop_action(env_state, False)

        a_opp = jax.lax.stop_gradient(a_opp_coop)
        a_opp = jnp.where(prev_agent_coin_collected_same_col == 0, a_opp_defect, a_opp)

    a1 = a
    a2 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, a, prev_agent_coin_collected_same_col, r1, r2)
    aux = (score1, score2)

    return stuff, aux


@jit
def eval_vs_tft_selfagent2(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, prev_a, prev_agent_coin_collected_same_col, r1, r2 = stuff

    key, subkey = jax.random.split(key)

    stuff, aux = generate_action(subkey, obsv, trainstate_th, trainstate_th.params, 
                     trainstate_val, trainstate_val.params, h_p, h_v)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    if args.env == "ipd":
        # Copy last move of agent; assumes prev_a = all coop
        a_opp = prev_a
        prev_agent_coin_collected_same_col = None
    elif args.env == "coin":

        r_opp = r1
        # Agent here means me, the agent we are testing
        prev_agent_coin_collected_same_col = jnp.where(r_opp < 0, 0, prev_agent_coin_collected_same_col)
        prev_agent_coin_collected_same_col = jnp.where(r_opp > 0, 1, prev_agent_coin_collected_same_col)

        a_opp_defect = env.get_moves_shortest_path_to_coin(env_state, True)
        a_opp_coop = env.get_coop_action(env_state, True)

        a_opp = jax.lax.stop_gradient(a_opp_coop)
        a_opp = jnp.where(prev_agent_coin_collected_same_col == 0, a_opp_defect, a_opp)

    a1 = a_opp
    a2 = a

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, a, prev_agent_coin_collected_same_col, r1, r2)
    aux = (score1, score2)

    return stuff, aux



@partial(jit, static_argnums=(3, 4))
def eval_vs_fixed_strategy(key, trainstate_th, trainstate_val, strat="alld", self_agent=1):

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    env_state, obsv = vec_env_reset(env_subkeys) # note this works only with the same obs, otherwise you would have to switch things up a bit here

    h_p = jnp.zeros((args.batch_size, args.hidden_size))
    h_v = None
    if use_baseline:
        h_v = jnp.zeros((args.batch_size, args.hidden_size))

    if strat == "alld":
        stuff = key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v

        if self_agent == 1:
            stuff, aux = jax.lax.scan(eval_vs_alld_selfagent1, stuff, None, args.rollout_len)
        else:
            stuff, aux = jax.lax.scan(eval_vs_alld_selfagent2, stuff, None, args.rollout_len)
    elif strat == "allc":
        stuff = key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v

        if self_agent == 1:
            stuff, aux = jax.lax.scan(eval_vs_allc_selfagent1, stuff, None, args.rollout_len)
        else:
            stuff, aux = jax.lax.scan(eval_vs_allc_selfagent2, stuff, None, args.rollout_len)
    elif strat == "tft":
        if args.env == "ipd":
            prev_a = jnp.ones(
                args.batch_size, dtype=int)  # assume agent (self) cooperated for the init time step when the opponent is using TFT
            r1 = jnp.zeros(args.batch_size)  # these don't matter for IPD,
            r2 = jnp.zeros(args.batch_size)
            prev_agent_coin_collected_same_col = None
        elif args.env == "coin":
            if self_agent == 1:
                prev_a = env.get_coop_action(env_state,
                                             red_agent_perspective=False)  # doesn't matter for coin
            else:
                prev_a = env.get_coop_action(env_state,
                                             red_agent_perspective=True)  # doesn't matter for coin
            prev_agent_coin_collected_same_col = jnp.ones(
                args.batch_size, dtype=int)  # 0 = defect, collect other agent coin. Init with 1 (coop)
            r1 = jnp.zeros(args.batch_size)
            r2 = jnp.zeros(args.batch_size)
        else:
            raise NotImplementedError
        stuff = (
        key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, prev_a,
        prev_agent_coin_collected_same_col, r1, r2)
        if self_agent == 1:
            stuff, aux = jax.lax.scan(eval_vs_tft_selfagent1, stuff, None,
                                      args.rollout_len)
        else:
            stuff, aux = jax.lax.scan(eval_vs_tft_selfagent2, stuff, None,
                                      args.rollout_len)

    score1, score2 = aux
    score1 = score1.mean()
    score2 = score2.mean()

    return (score1, score2), None

@jit
def get_init_hidden_states():
    h_p1, h_p2 = (
        jnp.zeros((args.batch_size, args.hidden_size)),
        jnp.zeros((args.batch_size, args.hidden_size))
    )
    h_v1, h_v2 = None, None
    if use_baseline:
        h_v1, h_v2 = (
            jnp.zeros((args.batch_size, args.hidden_size)),
            jnp.zeros((args.batch_size, args.hidden_size))
        )
    return h_p1, h_p2, h_v1, h_v2


def inspect_ipd(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2):
    assert args.env == 'ipd'
    unused_keys = jax.random.split(jax.random.PRNGKey(0), args.batch_size)
    state, obsv = vec_env_reset(unused_keys)

    init_state = env.init_state

    for i in range(2):
        for j in range(2):
            state1 = env.states[i, j]
            for ii in range(2):
                for jj in range(2):
                    state2 = env.states[ii, jj]

                    state_history = [init_state, state1, state2]
                    print(state_history)

                    pol_probs1 = get_policies_for_states_onebatch(jax.random.PRNGKey(0),
                                                         trainstate_th1,
                                                         trainstate_th1.params,
                                                         trainstate_val1,
                                                         trainstate_val1.params,
                                                         state_history)
                    pol_probs2 = get_policies_for_states_onebatch(jax.random.PRNGKey(0),
                                                         trainstate_th2,
                                                         trainstate_th2.params,
                                                         trainstate_val2,
                                                         trainstate_val2.params,
                                                         state_history)
                    print(pol_probs1)
                    print(pol_probs2)

    # Build state history artificially for all combs, and pass those into the pol_probs.





@jit
def eval_progress(subkey, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2):
    """
    Evaluate the performance of two agents in an environment.
    
    Parameters:
        subkey (jax.random.PRNGKey): The random key for generating subkeys.
        trainstate_th1 (TrainState), trainstate_val1 (TrainState): Training states for agent 1's policy and value function.
        trainstate_th2 (TrainState), trainstate_val2 (TrainState): Training states for agent 2's policy and value function.
        
    Returns:
        tuple: Various metrics like average rewards, match amounts, and scores against fixed strategies.
    """
    
    # Splitting the random key for environment and agents
    keys = jax.random.split(subkey, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]
    
    # Initialize environments and get initial observations and hidden states
    env_state, obs = vec_env_reset(env_subkeys)
    obs1, obs2 = obs, obs
    h_p1, h_p2, h_v1, h_v2 = get_init_hidden_states()
    
    # Prepare the initial state for the environment loop
    initial_state = (
        key, env_state, obs1, obs2,
        trainstate_th1, trainstate_th1.params, trainstate_val1, trainstate_val1.params,
        trainstate_th2, trainstate_th2.params, trainstate_val2, trainstate_val2.params,
        h_p1, h_v1, h_p2, h_v2
    )
    
    # Run the environment loop to collect rewards
    final_state, aux = jax.lax.scan(env_step, initial_state, None, args.rollout_len)
    aux1, aux2, aux_info = aux
    _, _, _, _, _, rewards1, _, _ = aux1
    _, _, _, _, _, rewards2, _, _ = aux2
    
    # Evaluate against fixed strategies
    scores1 = evaluate_against_strategies(key, trainstate_th1, trainstate_val1, self_agent=1)
    scores2 = evaluate_against_strategies(key, trainstate_th2, trainstate_val2, self_agent=2)
    
    # Compute average rewards
    avg_reward1 = rewards1.mean()
    avg_reward2 = rewards2.mean()
    
    if args.env == 'coin':
        rr, rb, br, bb = aux_info
        rr_count = rr.sum(axis=0).mean()
        rb_count = rb.sum(axis=0).mean()
        br_count = br.sum(axis=0).mean()
        bb_count = bb.sum(axis=0).mean()
        return avg_reward1, avg_reward2, rr_count, rb_count, br_count, bb_count, scores1, scores2

    return avg_reward1, avg_reward2, None, None, None, None, scores1, scores2


def evaluate_against_strategies(key, trainstate_policy, trainstate_value, self_agent):
    """
    Evaluate an agent's performance against fixed strategies.
    
    Parameters:
        key (jax.random.PRNGKey): The random key for generating subkeys.
        trainstate_policy (TrainState): Training state for the agent's policy.
        trainstate_value (TrainState): Training state for the agent's value function.
        self_agent (int): The index of the self agent (1 or 2).
        
    Returns:
        jnp.ndarray: Scores against fixed strategies.
    """
    
    scores = []
    for strategy in ["alld", "allc", "tft"]:
        key, subkey = jax.random.split(key)
        score, _ = eval_vs_fixed_strategy(subkey, trainstate_policy, trainstate_value, strategy, self_agent=self_agent)
        scores.append(score[self_agent - 1])
    return jnp.stack(scores)


def get_init_trainstates(key, action_size, input_size):
    """
    Initialize the training states for two agents.
    
    Args:
    - key: Random seed.
    - action_size: Number of possible actions.
    - input_size: Size of the input.
    
    Returns:
    - Tuple of training states for the two agents.
    """
    hidden_size = args.hidden_size
    key, key_p1, key_v1, key_p2, key_v2 = jax.random.split(key, 5)

    # Initialize RNNs for the two agents
    def init_rnn(key_p, key_v):
        theta_p = RNN(num_outputs=action_size,
                      num_hidden_units=hidden_size,
                      layers_before_gru=args.layers_before_gru)
        theta_v = RNN(num_outputs=1, num_hidden_units=hidden_size,
                      layers_before_gru=args.layers_before_gru)

        theta_p_params = theta_p.init(key_p, jnp.ones([args.batch_size, input_size]), jnp.zeros(hidden_size))
        theta_v_params = theta_v.init(key_v, jnp.ones([args.batch_size, input_size]), jnp.zeros(hidden_size))

        return theta_p, theta_p_params, theta_v, theta_v_params

    theta_p1, theta_p1_params, theta_v1, theta_v1_params = init_rnn(key_p1, key_v1)
    theta_p2, theta_p2_params, theta_v2, theta_v2_params = init_rnn(key_p2, key_v2)

    # Choose optimizer
    optimizers = {
        'adam': optax.adam(learning_rate=args.lr_out),
        'sgd': optax.sgd(learning_rate=args.lr_out)
    }
    theta_optimizer = optimizers.get(args.optim.lower())
    value_optimizer = optimizers.get(args.optim.lower(), optax.sgd(learning_rate=args.lr_v))

    if not theta_optimizer:
        raise Exception("Unknown or Not Implemented Optimizer")

    # Create training states
    trainstate_th1 = TrainState.create(apply_fn=theta_p1.apply, params=theta_p1_params, tx=theta_optimizer)
    trainstate_val1 = TrainState.create(apply_fn=theta_v1.apply, params=theta_v1_params, tx=value_optimizer)
    trainstate_th2 = TrainState.create(apply_fn=theta_p2.apply, params=theta_p2_params, tx=theta_optimizer)
    trainstate_val2 = TrainState.create(apply_fn=theta_v2.apply, params=theta_v2_params, tx=value_optimizer)

    return trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2



@jit
def get_c_e_for_om(key, om_trainstate_th, om_trainstate_th_params, om_trainstate_val, om_trainstate_val_params, other_state_history, other_act_history):
    key, subkey = jax.random.split(key)
    curr_pol_probs, h_p_list, h_v_list = get_policies_and_h_for_states(subkey, om_trainstate_th,
                                             om_trainstate_th_params,
                                             om_trainstate_val,
                                             om_trainstate_val_params,
                                             other_state_history)
    # KL div: p log p - p log q
    # use p for target, since it has 0 and 1
    # Then p log p has no deriv so can drop it, with respect to model
    # then -p log q

    # Calculate targets based on the action history (other act history)
    # Essentially treat the one hot vector of actions as a class label, and then run supervised learning

    c_e_loss = - (other_act_history * jnp.log(curr_pol_probs)).sum(
        axis=-1).mean()


    return c_e_loss, (h_p_list, h_v_list)

# This is just the same value loss as normal
# For OM we are assuming ability to observe other agents' actions and rewards (or alternatively, you assume the rewards are symmetrical to yours)
@jit
def get_val_loss_for_om(key, om_trainstate_th, om_trainstate_th_params, om_trainstate_val, om_trainstate_val_params,
                        other_state_history, other_act_history, rewards, end_state_v):
    key, subkey = jax.random.split(key)
    curr_pol_probs, curr_vals = get_policies_and_values_for_states(subkey, om_trainstate_th,
                                             om_trainstate_th_params,
                                             om_trainstate_val,
                                             om_trainstate_val_params,
                                             other_state_history)
    val_loss = value_loss(rewards, curr_vals, end_state_v)

    return val_loss

@jit
def opp_model_selfagent1_single_batch(inputstuff, unused ):
    key, trainstate_th1, trainstate_val1, true_other_trainstate_th, true_other_trainstate_val, om_trainstate_th, om_trainstate_val = inputstuff
    key, subkey = jax.random.split(key)

    stuff, aux, unfinished_state_history = do_env_rollout(subkey,
                                                          trainstate_th1,
                                                          trainstate_th1.params,
                                                          trainstate_val1,
                                                          trainstate_val1.params,
                                                          true_other_trainstate_th,
                                                          true_other_trainstate_th.params,
                                                          true_other_trainstate_val,
                                                          true_other_trainstate_val.params,
                                                          agent_for_state_history=2)

    key, env_state, obs1, obs2, \
    _, _, _, _, \
    _, _, _, _, \
    h_p1, h_v1, h_p2, h_v2 = stuff

    aux1, aux2, aux_info = aux

    cat_act_probs2_list, obs2_list, lp2_list, lp1_list, v2_list, r2_list, a2_list, a1_list = aux2

    unfinished_state_history.extend(obs2_list)
    other_state_history = unfinished_state_history

    other_act_history = a2_list
    other_rew_history = r2_list

    # I can do multiple "batches"
    # where repeating the below would be the same as collecting one big batch of environment interaction

    other_act_history = jax.nn.one_hot(other_act_history, action_size)

    om_grad_fn = jax.grad(get_c_e_for_om, argnums=2, has_aux=True)
    if use_baseline:
        om_val_grad_fn = jax.grad(get_val_loss_for_om, argnums=4)

    # This repeats training on one batch (one set of rollouts)
    for opp_model_iter in range(args.opp_model_steps_per_batch):

        key, subkey = jax.random.split(key)
        grad_th, h_lists = om_grad_fn(subkey, om_trainstate_th, om_trainstate_th.params,
                             om_trainstate_val, om_trainstate_val.params,
                             other_state_history, other_act_history)
        h_p_list, h_v_list = h_lists

        om_trainstate_th = om_trainstate_th.apply_gradients(grads=grad_th)


        if use_baseline:
            # act just to get the final state values
            key, subkey = jax.random.split(key)

            # For OM we should not be using the hidden states of the other agent's RNN
            # This only affects the generate_OM value function though, so shouldn't make a huge difference in terms of the actual policies learned.

            stuff2, aux2 = generate_action(subkey, obs2, om_trainstate_th, om_trainstate_th.params,
                               om_trainstate_val, om_trainstate_val.params, h_p_list[-1], h_v_list[-1], None)
            a2, lp2, v2, h_p2, h_v2, cat_act_probs2, logits2 = aux2

            end_state_v = v2
            grad_v = om_val_grad_fn(subkey, om_trainstate_th,
                                    om_trainstate_th.params, om_trainstate_val,
                                    om_trainstate_val.params,
                                    other_state_history, other_act_history,
                                    other_rew_history, end_state_v)

            om_trainstate_val = om_trainstate_val.apply_gradients(
                grads=grad_v)

    inputstuff = (key, trainstate_th1, trainstate_val1, true_other_trainstate_th, true_other_trainstate_val, om_trainstate_th, om_trainstate_val)
    aux = None
    return inputstuff, aux

@jit
def opp_model_selfagent2_single_batch(inputstuff, unused ):
    key, true_other_trainstate_th, true_other_trainstate_val, trainstate_th2, trainstate_val2, om_trainstate_th, om_trainstate_val = inputstuff

    key, subkey = jax.random.split(key)

    stuff, aux, unfinished_state_history = do_env_rollout(subkey,
                                                          true_other_trainstate_th,
                                                          true_other_trainstate_th.params,
                                                          true_other_trainstate_val,
                                                          true_other_trainstate_val.params,
                                                          trainstate_th2,
                                                          trainstate_th2.params,
                                                          trainstate_val2,
                                                          trainstate_val2.params,
                                                          agent_for_state_history=1)

    key, env_state, obs1, obs2, \
    _, _, _, _, \
    _, _, _, _, \
    h_p1, h_v1, h_p2, h_v2 = stuff

    aux1, aux2, aux_info = aux

    cat_act_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1

    unfinished_state_history.extend(obs1_list)
    other_state_history = unfinished_state_history

    other_act_history = a1_list
    other_rew_history = r1_list

    # I can do multiple "batches"
    # where repeating the below would be the same as collecting one big batch of environment interaction

    other_act_history = jax.nn.one_hot(other_act_history, action_size)

    om_grad_fn = jax.grad(get_c_e_for_om, argnums=2, has_aux=True)
    if use_baseline:
        om_val_grad_fn = jax.grad(get_val_loss_for_om, argnums=4)

    for opp_model_iter in range(args.opp_model_steps_per_batch):

        key, subkey = jax.random.split(key)
        grad_th, h_lists = om_grad_fn(subkey, om_trainstate_th, om_trainstate_th.params,
                             om_trainstate_val, om_trainstate_val.params,
                             other_state_history, other_act_history)
        h_p_list, h_v_list = h_lists

        om_trainstate_th = om_trainstate_th.apply_gradients(grads=grad_th)

        if use_baseline:
            # act just to get the final state values
            key, subkey = jax.random.split(key)

            stuff1, aux1 = generate_action(subkey, obs1, om_trainstate_th, om_trainstate_th.params,
                               om_trainstate_val, om_trainstate_val.params, h_p_list[-1], h_v_list[-1])

            a1, lp1, v1, h_p1, h_v1, cat_act_probs1, logits1 = aux1

            end_state_v = v1
            grad_v = om_val_grad_fn(subkey, om_trainstate_th,
                                    om_trainstate_th.params, om_trainstate_val,
                                    om_trainstate_val.params,
                                    other_state_history, other_act_history,
                                    other_rew_history, end_state_v)

            om_trainstate_val = om_trainstate_val.apply_gradients(
                grads=grad_v)

    inputstuff = (key, true_other_trainstate_th, true_other_trainstate_val, trainstate_th2, trainstate_val2, om_trainstate_th, om_trainstate_val)
    aux = None
    return inputstuff, aux



@jit
def opp_model_selfagent1(key, trainstate_th1, trainstate_val1, true_other_trainstate_th, true_other_trainstate_val,
              prev_om_trainstate_th, prev_om_trainstate_val):
    # true_other_theta_p and true_other_theta_v used only in the collection of data (rollouts in the environment)
    # so then this is not cheating. We do not assume access to other agent policy parameters (at least not direct, white box access)
    # We assume ability to collect trajectories through rollouts/play with the other agent in the environment
    # Essentially when using OM, we are now no longer doing dice update on the trajectories collected directly (which requires parameter access)
    # instead we collect the trajectories first, then build an OM, then rollout using OM and make DiCE/LOLA/POLA update based on that OM
    # Instead of direct rollout using opponent true parameters and update based on that.

    # not sure why I created copies of the om instead of directly using it but I don't think it hurts...
    # TODO May 31; try a few epochs with no copy and see if it's the same

    # Here have prev_om trainstates be the get_init_trainstates on the first iter before the first opp model
    om_trainstate_th = TrainState.create(apply_fn=prev_om_trainstate_th.apply_fn,
                                       params=prev_om_trainstate_th.params,
                                       tx=prev_om_trainstate_th.tx)
    om_trainstate_val = TrainState.create(apply_fn=prev_om_trainstate_val.apply_fn,
                                       params=prev_om_trainstate_val.params,
                                       tx=prev_om_trainstate_val.tx)
    key, subkey = jax.random.split(key)
    stuff = (subkey, trainstate_th1, trainstate_val1, true_other_trainstate_th, true_other_trainstate_val, om_trainstate_th, om_trainstate_val)
    stuff, aux = jax.lax.scan(opp_model_selfagent1_single_batch, stuff, None, args.opp_model_data_batches)
    _, trainstate_th1, trainstate_val1, true_other_trainstate_th, true_other_trainstate_val, om_trainstate_th, om_trainstate_val = stuff

    return om_trainstate_th, om_trainstate_val



@jit
def opp_model_selfagent2(key, true_other_trainstate_th, true_other_trainstate_val, trainstate_th2, trainstate_val2,
              prev_om_trainstate_th, prev_om_trainstate_val):
    # true_other_theta_p and true_other_theta_v used only in the collection of data (rollouts in the environment)
    # so then this is not cheating. We do not assume access to other agent policy parameters (at least not direct, white box access)
    # We assume ability to collect trajectories through rollouts/play with the other agent in the environment
    # Essentially when using OM, we are now no longer doing dice update on the trajectories collected directly (which requires parameter access)
    # instead we collect the trajectories first, then build an OM, then rollout using OM and make DiCE/LOLA/POLA update based on that OM
    # Instead of direct rollout using opponent true parameters and update based on that.

    # Here have prev_om trainstates be the get_init_trainstates on the first iter before the first opp model
    om_trainstate_th = TrainState.create(apply_fn=prev_om_trainstate_th.apply_fn,
                                       params=prev_om_trainstate_th.params,
                                       tx=prev_om_trainstate_th.tx)
    om_trainstate_val = TrainState.create(apply_fn=prev_om_trainstate_val.apply_fn,
                                       params=prev_om_trainstate_val.params,
                                       tx=prev_om_trainstate_val.tx)
    key, subkey = jax.random.split(key)
    stuff = (subkey, true_other_trainstate_th, true_other_trainstate_val, trainstate_th2, trainstate_val2, om_trainstate_th, om_trainstate_val)
    stuff, aux = jax.lax.scan(opp_model_selfagent2_single_batch, stuff, None, args.opp_model_data_batches)
    _, _, _, _, _, om_trainstate_th, om_trainstate_val = stuff

    return om_trainstate_th, om_trainstate_val


######################################## HELPER FUNCTIONS FOR PLAY ######################################################

def copyTrainState(trainstate):
    return TrainState.create(
        apply_fn=trainstate.apply_fn,
        params=trainstate.params,
        tx=trainstate.tx
    )

########################################################################################################################

def play(key, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2, use_opp_model=False):
    joint_scores = []  # combined scores of both agents after each update
    score_record = []  # individual scores of agents after each update
    vs_fixed_strategies_scores = [[], []]  # scores of agents against fixed strategies
    same_colour_coins_record = []  # counts of same-color coins collected in the "coin" environment
    diff_colour_coins_record = []  # counts of different-color coins collected in the "coin" environment
    coins_collected_info = (same_colour_coins_record, diff_colour_coins_record)


    print("start iterations with", args.inner_steps, "inner steps and", args.outer_steps, "outer steps:")


    if args.opp_model:
        key, subkey = jax.random.split(key)
        agent1_om_of_th2, agent1_om_of_val2, agent2_om_of_th1, agent2_om_of_val1 = get_init_trainstates(subkey, action_size, input_size)


    key, subkey = jax.random.split(key)
    score1, score2, rr_matches_amount, rb_matches_amount, br_matches_amount, bb_matches_amount, score1rec, score2rec = \
        eval_progress(key, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2)

    if args.env == "coin":
        same_colour_coins = rr_matches_amount + bb_matches_amount
        diff_colour_coins = rb_matches_amount + br_matches_amount
        same_colour_coins_record.append(same_colour_coins)
        diff_colour_coins_record.append(diff_colour_coins)

    vs_fixed_strategies_scores[0].append(score1rec)
    vs_fixed_strategies_scores[1].append(score2rec)

    score_record.append(jnp.stack((score1, score2)))


    for update in range(args.n_update):
        # THESE SHOULD NOT BE UPDATED (they are reset only on each new update step e.g. epoch, after all the outer and inner steps)
        trainstate_th1_ref, trainstate_val1_ref = copyTrainState(trainstate_th1), copyTrainState(trainstate_val1)
        trainstate_th2_ref, trainstate_val2_ref = copyTrainState(trainstate_th2), copyTrainState(trainstate_val2)


        # --- AGENT 1 UPDATE ---

        trainstate_th1_copy = copyTrainState(trainstate_th1)
        trainstate_val1_copy = copyTrainState(trainstate_val1)
        trainstate_th2_copy = copyTrainState(trainstate_th2)
        trainstate_val2_copy = copyTrainState(trainstate_val2)

        if args.opp_model:
            key, subkey = jax.random.split(key)
            agent1_om_of_th2, agent1_om_of_val2 = opp_model_selfagent1(subkey, trainstate_th1_copy, trainstate_val1_copy,
                                 trainstate_th2_copy, trainstate_val2_copy, agent1_om_of_th2, agent1_om_of_val2)
            # No need to overwrite the refs for agent 2 because those aren't used in the outer loop as we're using KL div for agent 1
            # The inner KL div is done in the inner loop which will automatically recreate/save the ref before each set of inner loop steps
            trainstate_th2_copy = copyTrainState(agent1_om_of_th2)
            trainstate_val2_copy = copyTrainState(agent1_om_of_val2)

        key, subkey = jax.random.split(key)

        stuff = (subkey, trainstate_th1_copy, trainstate_val1_copy,
                 trainstate_th2_copy, trainstate_val2_copy,
                 trainstate_th1_ref, trainstate_val1_ref)

        stuff, aux = jax.lax.scan(one_outer_step_update_selfagent1, stuff, None, args.outer_steps)
        _, trainstate_th1_copy, trainstate_val1_copy, _, _, _, _ = stuff

        trainstate_after_outer_steps_th1 = copyTrainState(trainstate_th1_copy)
        trainstate_after_outer_steps_val1 = copyTrainState(trainstate_val1_copy)

        # --- START OF AGENT 2 UPDATE ---

        # Doing this just as a safety failcase scenario, to make sure each agent loop starts from the beginning
        trainstate_th1_copy, trainstate_val1_copy = copyTrainState(trainstate_th1), copyTrainState(trainstate_val1)
        trainstate_th2_copy, trainstate_val2_copy = copyTrainState(trainstate_th2), copyTrainState(trainstate_val2)


        if args.opp_model:
            key, subkey = jax.random.split(key)
            agent2_om_of_th1, agent2_om_of_val1 = opp_model_selfagent2(subkey, trainstate_th1_copy, trainstate_val1_copy,
                                 trainstate_th2_copy, trainstate_val2_copy, agent2_om_of_th1, agent2_om_of_val1)
            # No need to overwrite the refs for agent 1 because those aren't used in the outer loop as we're using KL div for agent 2
            # The inner KL div is done in the inner loop which will automatically recreate/save the ref before each set of inner loop steps
            trainstate_th1_copy = copyTrainState(agent2_om_of_th1)
            trainstate_val1_copy = copyTrainState(agent2_om_of_val1)


        key, subkey = jax.random.split(key)

        stuff = (subkey, trainstate_th1_copy, trainstate_val1_copy,
                 trainstate_th2_copy, trainstate_val2_copy,
                 trainstate_th2_ref, trainstate_val2_ref)

        stuff, aux = jax.lax.scan(one_outer_step_update_selfagent2, stuff, None,
                                  args.outer_steps)
        _, _, _, trainstate_th2_copy, trainstate_val2_copy, _, _ = stuff

        trainstate_after_outer_steps_th2 = copyTrainState(trainstate_th2_copy)
        trainstate_after_outer_steps_val2 = copyTrainState(trainstate_val2_copy)

        # Note that this is updated only after all the outer loop steps have finished. the copies are
        # updated during the outer loops. But the main trainstate (like the main th) is updated only
        # after the loops finish
        trainstate_th1 = trainstate_after_outer_steps_th1
        trainstate_th2 = trainstate_after_outer_steps_th2

        trainstate_val1 = trainstate_after_outer_steps_val1
        trainstate_val2 = trainstate_after_outer_steps_val2


        # evaluate progress:
        key, subkey = jax.random.split(key)
        score1, score2, rr_matches_amount, rb_matches_amount, br_matches_amount, bb_matches_amount, score1rec, score2rec = \
            eval_progress(key, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2)



        if args.env == "coin":
            same_colour_coins = rr_matches_amount + bb_matches_amount
            diff_colour_coins = rb_matches_amount + br_matches_amount
            same_colour_coins_record.append(same_colour_coins)
            diff_colour_coins_record.append(diff_colour_coins)

        vs_fixed_strategies_scores[0].append(score1rec)
        vs_fixed_strategies_scores[1].append(score2rec)

        score_record.append(jnp.stack((score1, score2)))

        # print
        if (update + 1) % args.print_every == 0:
            print("*" * 10)
            print("Epoch: {}".format(update + 1), flush=True)
            print(f"Score for Agent 1: {score1}")
            print(f"Score for Agent 2: {score2}")
            if args.env == 'coin':
                print("Same coins: {}".format(rr_matches_amount + bb_matches_amount))
                print("Diff coins: {}".format(rb_matches_amount + br_matches_amount))
                print("RR coins {}".format(rr_matches_amount))
                print("RB coins {}".format(rb_matches_amount))
                print("BR coins {}".format(br_matches_amount))
                print("BB coins {}".format(bb_matches_amount))

            print("Scores vs fixed strats ALLD, ALLC, TFT:")
            print(score1rec)
            print(score2rec)

            if args.env == 'ipd' and args.inspect_ipd:
                    inspect_ipd(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2)

        if (update + 1) % args.checkpoint_every == 0:
            now = datetime.datetime.now()


            checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                        target=(trainstate_th1, trainstate_val1,
                                                trainstate_th2, trainstate_val2,
                                                coins_collected_info,
                                                score_record,
                                                vs_fixed_strategies_scores),
                                        step=update + 1, prefix=f"checkpoint_{now.strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch")


    return joint_scores




if __name__ == "__main__":
    parser = argparse.ArgumentParser("POLA")
    parser.add_argument("--inner_steps", type=int, default=1, help="inner loop steps for DiCE")
    parser.add_argument("--outer_steps", type=int, default=1, help="outer loop steps for POLA")
    parser.add_argument("--lr_out", type=float, default=0.005,
                        help="outer loop learning rate: same learning rate across all policies for now")
    parser.add_argument("--lr_in", type=float, default=0.03,
                        help="inner loop learning rate (eta): this has no use in the naive learning case. Used for the gradient step done for the lookahead for other agents during LOLA (therefore, often scaled to be higher than the outer learning rate in non-proximal LOLA). Note that this has a different meaning for the Taylor approx vs. actual update versions. A value of eta=1 is perfectly reasonable for the Taylor approx version as this balances the scale of the gradient with the naive learning term (and will be multiplied by the outer learning rate after), whereas for the actual update version with neural net, 1 is way too big an inner learning rate. For prox, this is the learning rate on the inner prox loop so is not that important - you want big enough to be fast-ish, but small enough to converge.")
    parser.add_argument("--lr_v", type=float, default=0.001,
                        help="same learning rate across all policies for now. Should be around maybe 0.001 or less for neural nets to avoid instability")
    parser.add_argument("--gamma", type=float, default=0.96, help="discount rate")
    parser.add_argument("--n_update", type=int, default=5000, help="number of epochs to run")
    parser.add_argument("--rollout_len", type=int, default=50, help="How long we want the time horizon of the game to be (number of steps before termination/number of iterations of the IPD)")
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1, help="for seed")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--print_every", type=int, default=1, help="Print every x number of epochs")
    parser.add_argument("--outer_beta", type=float, default=0.0, help="for outer kl penalty with POLA")
    parser.add_argument("--inner_beta", type=float, default=0.0, help="for inner kl penalty with POLA")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=50, help="Epochs between checkpoint save")
    parser.add_argument("--load_dir", type=str, default=None, help="Directory for loading checkpoint")
    parser.add_argument("--load_prefix", type=str, default=None, help="Prefix for loading checkpoint")
    parser.add_argument("--diff_coin_reward", type=float, default=1.0, help="changes problem setting (the reward for picking up coin of different colour)")
    parser.add_argument("--diff_coin_cost", type=float, default=-2.0, help="changes problem setting (the cost to the opponent when you pick up a coin of their colour)")
    parser.add_argument("--same_coin_reward", type=float, default=1.0, help="changes problem setting (the reward for picking up coin of same colour)")
    parser.add_argument("--grid_size", type=int, default=3, help="Grid size for Coin Game")
    parser.add_argument("--optim", type=str, default="adam", help="Used only for the outer agent (in the out_lookahead)")
    parser.add_argument("--no_baseline", action="store_true", help="Use NO Baseline (critic) for variance reduction. Default is baseline using Loaded DiCE with GAE")
    parser.add_argument("--opp_model", action="store_true", help="Use Opponent Modeling")
    parser.add_argument("--opp_model_steps_per_batch", type=int, default=1, help="How many steps to train opp model on each batch at the beginning of each POLA epoch")
    parser.add_argument("--opp_model_data_batches", type=int, default=100, help="How many batches of data (right now from rollouts) to train opp model on")
    parser.add_argument("--om_lr_p", type=float, default=0.005,
                        help="learning rate for opponent modeling (imitation/supervised learning) for policy")
    parser.add_argument("--om_lr_v", type=float, default=0.001,
                        help="learning rate for opponent modeling (imitation/supervised learning) for value")
    parser.add_argument("--env", type=str, default="coin",
                        choices=["ipd", "coin"])
    parser.add_argument("--hist_one", action="store_true", help="Use one step history (no gru or rnn, just one step history)")
    parser.add_argument("--print_info_each_outer_step", action="store_true", help="For debugging/curiosity sake")
    parser.add_argument("--init_state_coop", action="store_true", help="For IPD only: have the first state be CC instead of a separate start state")
    parser.add_argument("--split_coins", action="store_true", help="If true, then when both agents step on same coin, each gets 50% of the reward as if they were the only agent collecting that coin. Only tested with OGCoin so far")
    parser.add_argument("--zero_vals", action="store_true", help="For testing/debug. Can also serve as another way to do no_baseline. Set all values to be 0 in Loaded Dice Calculation")
    parser.add_argument("--gae_lambda", type=float, default=1,
                        help="lambda for GAE (1 = monte carlo style, 0 = TD style)")
    parser.add_argument("--val_update_after_loop", action="store_true", help="Update values only after outer POLA loop finishes, not during the POLA loop")
    parser.add_argument("--std", type=float, default=0.1, help="standard deviation for initialization of policy/value parameters")
    parser.add_argument("--inspect_ipd", action="store_true", help="Detailed (2 steps + start state) policy information in the IPD with full history")
    parser.add_argument("--layers_before_gru", type=int, default=2, choices=[0, 1, 2], help="Number of linear layers (with ReLU activation) before GRU, supported up to 2 for now")
    parser.add_argument("--contrib_factor", type=float, default=1.33, help="contribution factor to vary difficulty of IPD")
    parser.add_argument("--rev_kl", action="store_true", help="If true, then use KL(curr, target)")

    args = parser.parse_args()

    # Seed Initialization
    np.random.seed(args.seed)

    # Environment Setup
    if args.env == 'coin':
        assert args.grid_size == 3  # other sizes not implemented yet
        input_size = args.grid_size ** 2 * 4
        action_size = 4
        env = CoinGame()
    elif args.env == 'ipd':
        input_size = 6  # 3 * n_agents
        action_size = 2
        env = IPD(start_with_cooperation=args.init_state_coop, cooperation_factor=args.contrib_factor)
    else:
        raise NotImplementedError("unknown env")

    vec_env_reset = jax.vmap(env.reset)
    vec_env_step = jax.vmap(env.step)

    # Key Initialization
    key = jax.random.PRNGKey(args.seed)

    # Training State Initialization
    trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2 = get_init_trainstates(key, action_size, input_size)

    # Restore Checkpoints (if provided)
    if args.load_dir:
        epoch_num = int(args.load_prefix.split("epoch")[-1])
        
        # Temporary fix for updated checkpointing system
        if epoch_num % 10 == 0:
            epoch_num += 1

        # Initialize score records
        score_record = [jnp.zeros((2,))] * epoch_num
        vs_fixed_strategies_scores = [[jnp.zeros((3,))] * epoch_num, [jnp.zeros((3,))] * epoch_num]
        
        # Initialize coin collection records based on environment
        if args.env == 'coin':
            same_colour_coins_record = [jnp.zeros((1,))] * epoch_num
            diff_colour_coins_record = [jnp.zeros((1,))] * epoch_num
        else:
            same_colour_coins_record = []
            diff_colour_coins_record = []
        
        coins_collected_info = (same_colour_coins_record, diff_colour_coins_record)

        # Ensure a load prefix is provided
        assert args.load_prefix is not None
        
        # Restore checkpoint
        restored_tuple = checkpoints.restore_checkpoint(
            ckpt_dir=args.load_dir,
            target=(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
                    coins_collected_info, score_record, vs_fixed_strategies_scores),
            prefix=args.load_prefix
        )

        # Unpack restored tuple
        trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2, coins_collected_info, score_record, vs_fixed_strategies_scores = restored_tuple


    # Baseline Setup
    use_baseline = not args.no_baseline

    # Sanity Checks
    assert args.inner_steps >= 1
    assert args.outer_steps >= 1

    # Execute the Play Function
    joint_scores = play(key, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2, args.opp_model)

