import numpy as np
import argparse
import datetime

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


###############################################################################
#                               Utility Functions                             #
###############################################################################

def reverse_cumsum(x, axis):
    """
    Performs reverse cumulative sum along a given axis.
    Example:
      If x = [1, 2, 3], reverse_cumsum(x) would produce:
      [
        (1+2+3), (2+3), 3
      ]
    """
    return x + jnp.sum(x, axis=axis, keepdims=True) - jnp.cumsum(x, axis=axis)

@jit
def magic_box(x):
    """
    The DiCE "magic box" operator: exp(x - stop_gradient(x)).
    """
    return jnp.exp(x - jax.lax.stop_gradient(x))

@jit
def update_gae_with_delta_backwards(gae, delta):
    """
    Helper function for scan to accumulate generalized advantage estimates (GAE).
    """
    gae = gae * args.gamma * args.gae_lambda + delta
    return gae, gae

@jit
def get_gae_advantages(rewards, values, next_val_history):
    """
    Computes GAE advantages given rewards, value estimates, and next-step value estimates.
    """
    deltas = rewards + args.gamma * jax.lax.stop_gradient(next_val_history) - jax.lax.stop_gradient(values)
    gae_init = jnp.zeros_like(deltas[0, :])
    deltas_reversed = jnp.flip(deltas, axis=0)
    gae_final, flipped_advantages = jax.lax.scan(update_gae_with_delta_backwards, gae_init, deltas_reversed)
    advantages = jnp.flip(flipped_advantages, axis=0)
    return advantages


###############################################################################
#                          Core DiCE / GAE Objectives                         #
###############################################################################

@jit
def dice_objective(self_logprobs, other_logprobs, rewards, values, end_state_v):
    """
    Computes the (loaded) DiCE objective:
      If use_baseline is True, uses Loaded DiCE with GAE for variance reduction.
      If use_baseline is False, uses basic DiCE without baseline.
    """
    cum_discount = jnp.cumprod(args.gamma * jnp.ones(rewards.shape), axis=0) / args.gamma
    discounted_rewards = rewards * cum_discount

    # stochastics nodes involved in rewards dependencies:
    dependencies = jnp.cumsum(self_logprobs + other_logprobs, axis=0)

    # logprob of all stochastic nodes:
    stochastic_nodes = self_logprobs + other_logprobs

    use_loaded_dice = use_baseline

    if use_loaded_dice:
        next_val_history = jnp.zeros((args.rollout_len, args.batch_size))
        next_val_history = next_val_history.at[:args.rollout_len - 1, :].set(values[1:args.rollout_len, :])
        next_val_history = next_val_history.at[-1, :].set(end_state_v)

        if args.zero_vals:
            next_val_history = jnp.zeros_like(next_val_history)
            values = jnp.zeros_like(values)

        advantages = get_gae_advantages(rewards, values, next_val_history)
        discounted_advantages = advantages * cum_discount
        deps_up_to_t = jnp.cumsum(stochastic_nodes, axis=0)
        deps_less_than_t = deps_up_to_t - stochastic_nodes  # take out the dependency in the given time step

        # Formulation from Loaded DiCE and GAE papers
        loaded_dice_rewards = (magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * discounted_advantages
        dice_obj = loaded_dice_rewards.sum(axis=0).mean()
    else:
        dice_obj = jnp.mean(jnp.sum(magic_box(dependencies) * discounted_rewards, axis=0))

    return -dice_obj  # Minimizing negative of the objective.

@jit
def value_loss(rewards, values, final_state_vals):
    """
    Computes the MSE loss for the value function using a blend of Monte Carlo and bootstrapping.

    Specifically, it calculates a partial Monte Carlo return and incorporates a bootstrap
    estimate from the final state's value. This approach balances the bias-variance trade-off.
    """
    # Stop gradient on the final state values because these are target values for the value function.
    # We don't want to backpropagate through the value network when calculating the target.
    final_state_vals = jax.lax.stop_gradient(final_state_vals)

    # Calculate the discount factors for each time step.
    # 'args.gamma' is the discount rate.
    # We create an array of gamma values and compute the cumulative product to get gamma^t for each t.
    # Dividing by args.gamma ensures the first discount factor is 1.
    discounts = jnp.cumprod(args.gamma * jnp.ones(rewards.shape),
                                 axis=0) / args.gamma

    # Calculate the discounted rewards at each time step.
    gamma_t_r_ts = rewards * discounts

    # Calculate the sum of discounted rewards from each time step onwards, discounted to the first time step.
    # 'reverse_cumsum' computes the cumulative sum in reverse order.
    # The first entry of G_ts will contain the sum of all future discounted rewards.
    # The second entry will contain the sum of rewards from the second step onwards, discounted to the first time step, and so on.
    G_ts = reverse_cumsum(gamma_t_r_ts, axis=0)

    # Adjust the discounted rewards to be discounted to the appropriate current time step.
    # By dividing G_ts by the 'discounts', we effectively bring the discounted rewards to the correct time frame.
    # For example, after dividing by discounts, the rewards from time step 2 onwards are discounted only up to time step 2.
    R_ts = G_ts / discounts

    # Calculate the discounted value of the final state, discounted back to the current time steps.
    # 'jnp.flip(discounts, axis=0)' reverses the discounts array so that the discount factor aligns with the time remaining until the end.
    final_val_discounted_to_curr = (args.gamma * jnp.flip(discounts, axis=0)) * final_state_vals

    # Calculate the target values for the value function.
    # This is a mix of Monte Carlo return (R_ts) and a bootstrapped estimate from the final state value.
    # We add the discounted final state value to the Monte Carlo return.
    # It's crucial to detach the final state values (done above) because they serve as the target and should not contribute to the gradient of the value network in the current step.
    # This approach provides a more consistent value calculation, especially towards the end of an episode.
    target_values = R_ts + final_val_discounted_to_curr

    values_loss = (target_values - values) ** 2

    values_loss = values_loss.sum(axis=0).mean()

    return values_loss

@jit
def dice_objective_plus_value_loss(self_logprobs, other_logprobs, rewards, values, end_state_v):
    """
    This function serves as a wrapper to combine the DiCE objective and the value function loss.
    It ensures that gradients are computed correctly for both policy and value updates.
    The reward loss has a stop_gradient on nodes related to the value function, and the value
    function loss has no nodes related to the policy. This allows for independent updates.
    If use_baseline is True, it sums the DiCE objective with the value function loss; otherwise,
    it returns only the DiCE objective.
    """
    reward_loss = dice_objective(self_logprobs, other_logprobs, rewards, values, end_state_v)
    if use_baseline:
        val_loss = value_loss(rewards, values, end_state_v)
        return reward_loss + val_loss
    else:
        return reward_loss



###############################################################################
#                           Acting in the Environment                         #
###############################################################################

@jit
def act(scan_carry, _):
    """
    Makes a single step action with a policy RNN:
    - sample the action from the categorical distribution,
    - compute log probability of the action,
    - optionally compute value if use_baseline is True.
    """
    (key, obs_batch, trainstate_p, trainstate_p_params,
     trainstate_v, trainstate_v_params, hidden_p, hidden_v) = scan_carry

    hidden_p, logits = trainstate_p.apply_fn(trainstate_p_params, obs_batch, hidden_p)
    dist = tfd.Categorical(logits=logits)
    key, subkey = jax.random.split(key)
    actions = dist.sample(seed=subkey)
    log_probs_actions = dist.log_prob(actions)

    if use_baseline:
        hidden_v, values = trainstate_v.apply_fn(trainstate_v_params, obs_batch, hidden_v)
        ret_vals = values.squeeze(-1)
    else:
        hidden_v, ret_vals = None, None

    new_scan_carry = (key, obs_batch, trainstate_p, trainstate_p_params,
                      trainstate_v, trainstate_v_params, hidden_p, hidden_v)
    auxiliary = (actions, log_probs_actions, ret_vals, hidden_p, hidden_v,
                 jax.nn.softmax(logits), logits)
    return new_scan_carry, auxiliary

@jit
def act_w_iter_over_obs(scan_carry, env_batch_obs):
    """
    jax.lax.scan wrapper to iterate over multiple observations in the environment 
    rollout and apply `act` at each step.
    """
    key, p_state, p_params, v_state, v_params, h_p, h_v = scan_carry
    key, subkey = jax.random.split(key)
    act_input = (subkey, env_batch_obs, p_state, p_params, v_state, v_params, h_p, h_v)
    new_act_input, act_aux = act(act_input, None)
    (_, _, p_state, p_params, v_state, v_params, h_p, h_v) = new_act_input
    new_scan_carry = (key, p_state, p_params, v_state, v_params, h_p, h_v)
    return new_scan_carry, act_aux

@jit
def env_step(scan_carry, _):
    """
    Single environment step for both agents (1 and 2). Each agent acts using 
    policy RNN, obtains next state, rewards, etc.
    """
    (key, env_state, obs1, obs2,
     th1, th1_params, val1, val1_params,
     th2, th2_params, val2, val2_params,
     h_p1, h_v1, h_p2, h_v2) = scan_carry

    key, key1, key2, env_key = jax.random.split(key, 4)

    # Agent 1 step
    act_args1 = (key1, obs1, th1, th1_params, val1, val1_params, h_p1, h_v1)
    _, aux1 = act(act_args1, None)
    a1, lp1, v1, h_p1, h_v1, cat_probs1, logits1 = aux1

    # Agent 2 step
    act_args2 = (key2, obs2, th2, th2_params, val2, val2_params, h_p2, h_v2)
    _, aux2 = act(act_args2, None)
    a2, lp2, v2, h_p2, h_v2, cat_probs2, logits2 = aux2

    env_subkeys = jax.random.split(env_key, args.batch_size)
    env_state_next, obs_next, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)

    scan_carry_next = (key, env_state_next, obs_next, obs_next,
                       th1, th1_params, val1, val1_params,
                       th2, th2_params, val2, val2_params,
                       h_p1, h_v1, h_p2, h_v2)

    # Aux for each agent is (policy distribution, obs, self_logprob, other_logprob, self_values, reward, self_action, other_action)
    aux1_out = (cat_probs1, obs_next, lp1, lp2, v1, r1, a1, a2)
    aux2_out = (cat_probs2, obs_next, lp2, lp1, v2, r2, a2, a1)

    return scan_carry_next, (aux1_out, aux2_out, aux_info)


###############################################################################
#                    Policy / Value RNN definitions (Flax)                    #
###############################################################################

class RNN(nn.Module):
    """
    A simple RNN using optional Dense layers (layers_before_gru) before 
    feeding into a GRUCell, then a final Dense to produce outputs.
    Only supports 2 layers before GRU for now.
    """
    num_outputs: int
    num_hidden_units: int
    layers_before_gru: int

    def setup(self):
        if self.layers_before_gru >= 1:
            self.linear1 = nn.Dense(features=self.num_hidden_units)
        if self.layers_before_gru >= 2:
            self.linear2 = nn.Dense(features=self.num_hidden_units)
        self.GRUCell = nn.GRUCell(features=self.num_hidden_units)
        self.linear_end = nn.Dense(features=self.num_outputs)

    def __call__(self, x, carry):
        if self.layers_before_gru >= 1:
            x = self.linear1(x)
            x = nn.relu(x)
        if self.layers_before_gru >= 2:
            x = self.linear2(x)

        carry, x = self.GRUCell(carry, x)
        outputs = self.linear_end(x)
        return carry, outputs


###############################################################################
#              Helpers: RNN forward calls for multiple time steps             #
###############################################################################

@jit
def get_policies_for_states(key, train_p, train_p_params, train_v, train_v_params, obs_hist):
    """
    Iterates over obs_hist with a policy RNN, returning the probability 
    distribution for each time step's observation. 
    Does not return values. 
    """
    h_p = jnp.zeros((args.batch_size, args.hidden_size))
    h_v = jnp.zeros((args.batch_size, args.hidden_size)) if use_baseline else None
    key, subkey = jax.random.split(key)
    init_scan_carry = (subkey, train_p, train_p_params, train_v, train_v_params, h_p, h_v)
    obs_hist_for_scan = jnp.stack(obs_hist[:args.rollout_len], axis=0) # skips the last observation
    final_scan_carry, aux_lists = jax.lax.scan(act_w_iter_over_obs, init_scan_carry, obs_hist_for_scan, args.rollout_len)
    (_, _, _, _, _, _, _), (a_list, lp_list, v_list, h_p_list, h_v_list, cat_act_probs_list, logits_list) = (final_scan_carry, aux_lists)
    return cat_act_probs_list

@jit
def get_policies_and_values_for_states(key, train_p, train_p_params, train_v, train_v_params, obs_hist):
    """
    Iterates over obs_hist with RNN, returning both probabilities and value estimates.
    """
    h_p = jnp.zeros((args.batch_size, args.hidden_size))
    h_v = jnp.zeros((args.batch_size, args.hidden_size)) if use_baseline else None
    key, subkey = jax.random.split(key)
    init_scan_carry = (subkey, train_p, train_p_params, train_v, train_v_params, h_p, h_v)
    obs_hist_for_scan = jnp.stack(obs_hist[:args.rollout_len], axis=0)
    final_scan_carry, aux_lists = jax.lax.scan(act_w_iter_over_obs, init_scan_carry, obs_hist_for_scan, args.rollout_len)
    (_, _, _, _, _, _, _), (a_list, lp_list, v_list, h_p_list, h_v_list, cat_probs_list, logits_list) = (final_scan_carry, aux_lists)
    return cat_probs_list, v_list

def get_init_hidden_states():
    """
    Returns initial hidden states for the policy & value RNNs.
    """
    h_p1 = jnp.zeros((args.batch_size, args.hidden_size))
    h_p2 = jnp.zeros((args.batch_size, args.hidden_size))
    h_v1 = None
    h_v2 = None
    if use_baseline:
        h_v1 = jnp.zeros((args.batch_size, args.hidden_size))
        h_v2 = jnp.zeros((args.batch_size, args.hidden_size))
    return h_p1, h_p2, h_v1, h_v2


###############################################################################
#              Environment Rollout Helpers (N-step scanning)                  #
###############################################################################

@partial(jit, static_argnums=(9))
def do_env_rollout(key, th1, th1_params, val1, val1_params, th2, th2_params, val2, val2_params, agent_for_state_history):
    """
    Performs a multi-step environment rollout for each of the batch_size parallel 
    envs using a vmap of step() calls. Saves partial state history for the 
    agent_of_interest (agent_for_state_history).
    """
    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]
    env_state, obsv = vec_env_reset(env_subkeys)
    obs1 = obsv
    obs2 = obsv
    h_p1, h_p2, h_v1, h_v2 = get_init_hidden_states()
    state_history = []
    if agent_for_state_history == 2:
        state_history.append(obs2)
    else:
        state_history.append(obs1)

    # state_history stores the initial observation for the agent of interest.
    # Additional observations from the rollout will be appended later. (strange but works)

    init_scan_carry = (key, env_state, obs1, obs2, th1, th1_params, val1, val1_params,
                       th2, th2_params, val2, val2_params, h_p1, h_v1, h_p2, h_v2)
    final_scan_carry, aux = jax.lax.scan(env_step, init_scan_carry, None, args.rollout_len)

    return final_scan_carry, aux, state_history


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


###############################################################################
#    Inner-Loop Minimization for Opponent's Policy (POLA Inner Step)          #
###############################################################################

@jit
def kl_div_jax(curr, target):
    """
    Forward KL:  KL(target || curr) = E_target[log(target) - log(curr)].
    """
    return (target * (jnp.log(target) - jnp.log(curr))).sum(axis=-1).mean()

@jit
def rev_kl_div_jax(curr, target):
    """
    Reverse KL:  KL(curr || target) = E_curr[log(curr) - log(target)].
    """
    return (curr * (jnp.log(curr) - jnp.log(target))).sum(axis=-1).mean()

@partial(jit, static_argnums=(11))
def in_lookahead(key, th1, th1_params, val1, val1_params,
                 th2, th2_params, val2, val2_params,
                 old_trainstate_th, old_trainstate_val, other_agent=2):
    """
    The "inner" lookahead for the specified agent (other_agent).
    We do a single rollout with the current policies, compute the agent-of-interest's 
    objective, and add a KL penalty with old parameters for a proximal step.
    """
    final_carry, aux_data, state_hist = do_env_rollout(key, th1, th1_params, val1, val1_params,
                                                       th2, th2_params, val2, val2_params,
                                                       agent_for_state_history=other_agent)
    aux1, aux2, aux_info = aux_data

    (key_final, env_state, obs1, obs2,
     _, _, _, _, _, _, _, _, h_p1, h_v1, h_p2, h_v2) = final_carry

    # Depending on who is "other_agent", parse out the correct rollout data
    if other_agent == 2:
        cat_probs2_list, obs2_list, lp2_list, lp1_list, v2_list, r2_list, a2_list, a1_list = aux2
        state_hist.extend(obs2_list)
        act_args2 = (key_final, obs2, th2, th2_params, val2, val2_params, h_p2, h_v2)
        _, (a2_end, lp2_end, v2_end, _, _, _, _) = act(act_args2, None)
        end_state_v2 = v2_end
        objective_inner = dice_objective_plus_value_loss(lp2_list, lp1_list, r2_list, v2_list, end_state_v2)
        key_next, subkey1 = jax.random.split(key_final, 2)
        pol_probs = get_policies_for_states(subkey1, th2, th2_params, val2, val2_params, state_hist)
        pol_probs_old = get_policies_for_states(subkey1, old_trainstate_th, old_trainstate_th.params,
                                                old_trainstate_val, old_trainstate_val.params,
                                                state_hist)
    else:
        cat_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1
        state_hist.extend(obs1_list)
        act_args1 = (key_final, obs1, th1, th1_params, val1, val1_params, h_p1, h_v1)
        _, (a1_end, lp1_end, v1_end, _, _, _, _) = act(act_args1, None)
        end_state_v1 = v1_end
        objective_inner = dice_objective_plus_value_loss(lp1_list, lp2_list, r1_list, v1_list, end_state_v1)
        key_next, subkey1 = jax.random.split(key_final, 2)
        pol_probs = get_policies_for_states(subkey1, th1, th1_params, val1, val1_params, state_hist)
        pol_probs_old = get_policies_for_states(subkey1, old_trainstate_th, old_trainstate_th.params,
                                                old_trainstate_val, old_trainstate_val.params,
                                                state_hist)

    # The current KL divergence is calculated using the state history of this episode,
    # passed through both the current and old policy parameters. This provides fresh
    # data for each inner step, potentially leading to more stable training and better
    # coverage of the state space compared to using a fixed initial batch. For repeated
    # training, KL divergence should be based on initial trajectory (save it and resuse in JAX).

    if args.rev_kl:
        kl_term = rev_kl_div_jax(pol_probs, pol_probs_old)
    else:
        kl_term = kl_div_jax(pol_probs, pol_probs_old)

    return objective_inner + args.inner_beta * kl_term

@jit
def inner_step_get_grad_otheragent2(scan_carry, _):
    """
    Single update step for agent 2's inner lookahead objective.
    """
    (key, th1, th1_params, val1, val1_params,
     th2, th2_params, val2, val2_params,
     old_th, old_val) = scan_carry

    key, subkey = jax.random.split(key)
    grad_fn = jax.grad(in_lookahead, argnums=[6, 8])
    grad_th2, grad_v2 = grad_fn(subkey, th1, th1_params, val1, val1_params,
                                th2, th2_params, val2, val2_params,
                                old_th, old_val, other_agent=2)
    # Update agent 2's policy parameters (SGD)
    th2_updated = th2.apply_gradients(grads=grad_th2)

    # Update agent 2's value parameters (SGD)
    val2_updated = val2.apply_gradients(grads=grad_v2) if use_baseline else val2

    new_scan_carry = (key, th1, th1_params, val1, val1_params,
                      th2_updated, th2_updated.params, val2_updated, val2_updated.params,
                      old_th, old_val)
    return new_scan_carry, None

@jit
def inner_step_get_grad_otheragent1(scan_carry, _):
    """
    Single update step for agent 1's inner lookahead objective.
    """
    (key, th1, th1_params, val1, val1_params,
     th2, th2_params, val2, val2_params,
     old_th, old_val) = scan_carry

    key, subkey = jax.random.split(key)
    grad_fn = jax.grad(in_lookahead, argnums=[2, 4])
    grad_th1, grad_v1 = grad_fn(subkey, th1, th1_params, val1, val1_params,
                                th2, th2_params, val2, val2_params,
                                old_th, old_val, other_agent=1)
    # Update agent 1's policy parameters (SGD)
    th1_updated = th1.apply_gradients(grads=grad_th1)

    # Update agent 1's value parameters (SGD)
    val1_updated = val1.apply_gradients(grads=grad_v1) if use_baseline else val1

    new_scan_carry = (key, th1_updated, th1_updated.params,
                      val1_updated, val1_updated.params,
                      th2, th2_params, val2, val2_params,
                      old_th, old_val)
    return new_scan_carry, None

@jit
def inner_steps_plus_update_otheragent2(key, th1, th1_params, val1, val1_params,
                                        th2, th2_params, val2, val2_params,
                                        old_th2, old_val2):
    """
    Runs args.inner_steps of agent 2's inner loop updates, returning updated (th2, val2).
    """
    th2_prime = TrainState.create(apply_fn=th2.apply_fn, params=th2_params,
                                  tx=optax.sgd(learning_rate=args.lr_in))
    val2_prime = None
    if use_baseline:
        val2_prime = TrainState.create(apply_fn=val2.apply_fn, params=val2_params,
                                       tx=optax.sgd(learning_rate=args.lr_v))
    else:
        val2_prime = val2

    # Save parameters of agent 1's network for differentiation in the outer loop.
    carry_init = (key, th1, th1_params, val1, val1_params,
                  th2_prime, th2_prime.params, val2_prime, val2_prime.params,
                  old_th2, old_val2)

    carry_after, _ = inner_step_get_grad_otheragent2(carry_init, None)
    (key_after, th1_, th1_params_, val1_, val1_params_,
     th2_prime, th2_prime_params, val2_prime, val2_prime_params,
     _, _) = carry_after

    if args.inner_steps > 1:
        carry_loop = (key_after, th1_, th1_params_, val1_, val1_params_,
                      th2_prime, th2_prime.params, val2_prime, val2_prime.params,
                      old_th2, old_val2)
        # Each step in the scan updates agent 2's policy and value network parameters using SGD.
        # Gradients are calculated based on the parameters from the previous iteration.
        carry_loop, _ = jax.lax.scan(inner_step_get_grad_otheragent2, carry_loop,
                                     None, args.inner_steps - 1)
        (_, _, _, _, _, th2_prime, th2_prime_params, val2_prime,
         val2_prime_params, _, _) = carry_loop

    return th2_prime, val2_prime

@jit
def inner_steps_plus_update_otheragent1(key, th1, th1_params, val1, val1_params,
                                        th2, th2_params, val2, val2_params,
                                        old_th1, old_val1):
    """
    Runs args.inner_steps of agent 1's inner loop updates, returning updated (th1, val1).
    """
    th1_prime = TrainState.create(apply_fn=th1.apply_fn, params=th1_params,
                                  tx=optax.sgd(learning_rate=args.lr_in))
    val1_prime = None
    if use_baseline:
        val1_prime = TrainState.create(apply_fn=val1.apply_fn, params=val1_params,
                                       tx=optax.sgd(learning_rate=args.lr_v))
    else:
        val1_prime = val1

    # Save parameters of agent 2's networks for differentiation in the outer loop.
    carry_init = (key, th1_prime, th1_prime.params,
                  val1_prime, val1_prime.params,
                  th2, th2_params, val2, val2_params, old_th1, old_val1)
    carry_after, _ = inner_step_get_grad_otheragent1(carry_init, None)
    (_, th1_prime, _, val1_prime, _, _, _, _, _, _, _) = carry_after

    if args.inner_steps > 1:
        carry_loop = (carry_after[0], th1_prime, th1_prime.params,
                      val1_prime, val1_prime.params,
                      th2, th2_params, val2, val2_params, old_th1, old_val1)
        # Each step in the scan updates agent 1's policy and value network parameters using SGD.
        # Gradients are calculated based on the parameters from the previous iteration.
        carry_loop, _ = jax.lax.scan(inner_step_get_grad_otheragent1, carry_loop,
                                     None, args.inner_steps - 1)
        (_, th1_prime, _, val1_prime, _, _, _, _, _, _, _) = carry_loop

    return th1_prime, val1_prime


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

        act_args1 = (subkey, obs1, trainstate_th1, trainstate_th1_params,
                     trainstate_val1, trainstate_val1_params, h_p1, h_v1)
        stuff1, aux1 = act(act_args1, None)
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
        act_args2 = (subkey, obs2, trainstate_th2, trainstate_th2_params,
                     trainstate_val2, trainstate_val2_params, h_p2, h_v2)
        stuff2, aux2 = act(act_args2, None)
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

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
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

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
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

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
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

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
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

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
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

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
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
    keys = jax.random.split(subkey, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]
    env_state, obsv = vec_env_reset(env_subkeys)
    obs1 = obsv
    obs2 = obsv
    h_p1, h_p2, h_v1, h_v2 = get_init_hidden_states()
    key, subkey = jax.random.split(key)
    stuff = (subkey, env_state, obs1, obs2,
             trainstate_th1, trainstate_th1.params, trainstate_val1,
             trainstate_val1.params,
             trainstate_th2, trainstate_th2.params, trainstate_val2,
             trainstate_val2.params,
             h_p1, h_v1, h_p2, h_v2)

    stuff, aux = jax.lax.scan(env_step, stuff, None, args.rollout_len)
    aux1, aux2, aux_info = aux

    _, _, _, _, _, r1, _, _ = aux1
    _, _, _, _, _, r2, _, _ = aux2

    score1rec = []
    score2rec = []

    print("Eval vs Fixed Strategies:")
    for strat in ["alld", "allc", "tft"]:
        # print(f"Playing against strategy: {strat.upper()}")
        key, subkey = jax.random.split(key)
        score1, _ = eval_vs_fixed_strategy(subkey, trainstate_th1, trainstate_val1, strat, self_agent=1)
        score1rec.append(score1[0])
        # print(f"Agent 1 score: {score1[0]}")
        key, subkey = jax.random.split(key)
        score2, _ = eval_vs_fixed_strategy(subkey, trainstate_th2, trainstate_val2, strat, self_agent=2)
        score2rec.append(score2[1])
        # print(f"Agent 2 score: {score2[1]}")

    score1rec = jnp.stack(score1rec)
    score2rec = jnp.stack(score2rec)

    avg_rew1 = r1.mean()
    avg_rew2 = r2.mean()

    if args.env == 'coin':
        rr_matches, rb_matches, br_matches, bb_matches = aux_info
        rr_matches_amount = rr_matches.sum(axis=0).mean()
        rb_matches_amount = rb_matches.sum(axis=0).mean()
        br_matches_amount = br_matches.sum(axis=0).mean()
        bb_matches_amount = bb_matches.sum(axis=0).mean()
        return avg_rew1, avg_rew2, rr_matches_amount, rb_matches_amount, br_matches_amount, bb_matches_amount, score1rec, score2rec

    else:
        return avg_rew1, avg_rew2, None, None, None, None, score1rec, score2rec


def get_init_trainstates(key, action_size, input_size):
    hidden_size = args.hidden_size

    key, key_p1, key_v1, key_p2, key_v2 = jax.random.split(key, 5)

    theta_p1 = RNN(num_outputs=action_size,
                   num_hidden_units=hidden_size,
                   layers_before_gru=args.layers_before_gru)
    theta_v1 = RNN(num_outputs=1, num_hidden_units=hidden_size,
                   layers_before_gru=args.layers_before_gru)

    theta_p1_params = theta_p1.init(key_p1, jnp.ones(
        [args.batch_size, input_size]), jnp.zeros(hidden_size))
    theta_v1_params = theta_v1.init(key_v1, jnp.ones(
        [args.batch_size, input_size]), jnp.zeros(hidden_size))

    theta_p2 = RNN(num_outputs=action_size,
                   num_hidden_units=hidden_size,
                   layers_before_gru=args.layers_before_gru)
    theta_v2 = RNN(num_outputs=1, num_hidden_units=hidden_size,
                   layers_before_gru=args.layers_before_gru)

    theta_p2_params = theta_p2.init(key_p2, jnp.ones(
        [args.batch_size, input_size]), jnp.zeros(hidden_size))
    theta_v2_params = theta_v2.init(key_v2, jnp.ones(
        [args.batch_size, input_size]), jnp.zeros(hidden_size))

    if args.optim.lower() == 'adam':
        theta_optimizer = optax.adam(learning_rate=args.lr_out)
        value_optimizer = optax.adam(learning_rate=args.lr_v)
    elif args.optim.lower() == 'sgd':
        theta_optimizer = optax.sgd(learning_rate=args.lr_out)
        value_optimizer = optax.sgd(learning_rate=args.lr_v)
    else:
        raise Exception("Unknown or Not Implemented Optimizer")

    trainstate_th1 = TrainState.create(apply_fn=theta_p1.apply,
                                       params=theta_p1_params,
                                       tx=theta_optimizer)
    trainstate_val1 = TrainState.create(apply_fn=theta_v1.apply,
                                        params=theta_v1_params,
                                        tx=value_optimizer)
    trainstate_th2 = TrainState.create(apply_fn=theta_p2.apply,
                                       params=theta_p2_params,
                                       tx=theta_optimizer)
    trainstate_val2 = TrainState.create(apply_fn=theta_v2.apply,
                                        params=theta_v2_params,
                                        tx=value_optimizer)

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
            # This only affects the OM value function though, so shouldn't make a huge difference in terms of the actual policies learned.
            act_args2 = (
                subkey, obs2, om_trainstate_th, om_trainstate_th.params,
                om_trainstate_val, om_trainstate_val.params, h_p_list[-1], h_v_list[-1])
            stuff2, aux2 = act(act_args2, None)
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
            act_args1 = (
                subkey, obs1, om_trainstate_th, om_trainstate_th.params,
                om_trainstate_val, om_trainstate_val.params, h_p_list[-1], h_v_list[-1])
            stuff1, aux1 = act(act_args1, None)
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



def play(key, init_trainstate_th1, init_trainstate_val1, init_trainstate_th2, init_trainstate_val2, use_opp_model=False):
    joint_scores = []
    score_record = []
    # You could do something like the below and then modify the code to just be one continuous record that includes past values when loading from checkpoint
    # if prev_scores is not None:
    #     score_record = prev_scores
    # I'm tired though.
    vs_fixed_strats_score_record = [[], []]

    print("start iterations with", args.inner_steps, "inner steps and", args.outer_steps, "outer steps:")
    same_colour_coins_record = []
    diff_colour_coins_record = []
    coins_collected_info = (same_colour_coins_record, diff_colour_coins_record)

    # Pretty sure this creation is unnecessary and we can directly use the trainstates passed in
    trainstate_th1 = TrainState.create(apply_fn=init_trainstate_th1.apply_fn,
                                       params=init_trainstate_th1.params,
                                       tx=init_trainstate_th1.tx)
    trainstate_val1 = TrainState.create(apply_fn=init_trainstate_val1.apply_fn,
                                        params=init_trainstate_val1.params,
                                        tx=init_trainstate_val1.tx)
    trainstate_th2 = TrainState.create(apply_fn=init_trainstate_th2.apply_fn,
                                       params=init_trainstate_th2.params,
                                       tx=init_trainstate_th2.tx)
    trainstate_val2 = TrainState.create(apply_fn=init_trainstate_val2.apply_fn,
                                        params=init_trainstate_val2.params,
                                        tx=init_trainstate_val2.tx)

    if args.opp_model:
        key, subkey = jax.random.split(key)
        agent1_om_of_th2, agent1_om_of_val2, agent2_om_of_th1, agent2_om_of_val1 = get_init_trainstates(subkey, action_size, input_size)


    key, subkey = jax.random.split(key)
    score1, score2, rr_matches_amount, rb_matches_amount, br_matches_amount, bb_matches_amount, score1rec, score2rec = \
        eval_progress(key, trainstate_th1, trainstate_val1, trainstate_th2,
                      trainstate_val2)

    if args.env == "coin":
        same_colour_coins = rr_matches_amount + bb_matches_amount
        diff_colour_coins = rb_matches_amount + br_matches_amount
        same_colour_coins_record.append(same_colour_coins)
        diff_colour_coins_record.append(diff_colour_coins)

    vs_fixed_strats_score_record[0].append(score1rec)
    vs_fixed_strats_score_record[1].append(score2rec)

    score_record.append(jnp.stack((score1, score2)))


    for update in range(args.n_update):
        # TODO there may be redundancy here (as in many places in this code...), consider clean up later
        # THESE SHOULD NOT BE UPDATED (they are reset only on each new update step e.g. epoch, after all the outer and inner steps)
        trainstate_th1_ref = TrainState.create(
            apply_fn=trainstate_th1.apply_fn,
            params=trainstate_th1.params,
            tx=trainstate_th1.tx)
        trainstate_val1_ref = TrainState.create(
            apply_fn=trainstate_val1.apply_fn,
            params=trainstate_val1.params,
            tx=trainstate_val1.tx)
        trainstate_th2_ref = TrainState.create(
            apply_fn=trainstate_th2.apply_fn,
            params=trainstate_th2.params,
            tx=trainstate_th2.tx)
        trainstate_val2_ref = TrainState.create(
            apply_fn=trainstate_val2.apply_fn,
            params=trainstate_val2.params,
            tx=trainstate_val2.tx)


        # --- AGENT 1 UPDATE ---

        trainstate_th1_copy = TrainState.create(
            apply_fn=trainstate_th1.apply_fn,
            params=trainstate_th1.params,
            tx=trainstate_th1.tx)
        trainstate_val1_copy = TrainState.create(
            apply_fn=trainstate_val1.apply_fn,
            params=trainstate_val1.params,
            tx=trainstate_val1.tx)
        trainstate_th2_copy = TrainState.create(
            apply_fn=trainstate_th2.apply_fn,
            params=trainstate_th2.params,
            tx=trainstate_th2.tx)
        trainstate_val2_copy = TrainState.create(
            apply_fn=trainstate_val2.apply_fn,
            params=trainstate_val2.params,
            tx=trainstate_val2.tx)

        if args.opp_model:
            key, subkey = jax.random.split(key)
            agent1_om_of_th2, agent1_om_of_val2 = opp_model_selfagent1(subkey, trainstate_th1_copy, trainstate_val1_copy,
                                 trainstate_th2_copy, trainstate_val2_copy, agent1_om_of_th2, agent1_om_of_val2)
            # No need to overwrite the refs for agent 2 because those aren't used in the outer loop as we're using KL div for agent 1
            # The inner KL div is done in the inner loop which will automatically recreate/save the ref before each set of inner loop steps
            trainstate_th2_copy = TrainState.create(
                apply_fn=agent1_om_of_th2.apply_fn,
                params=agent1_om_of_th2.params,
                tx=agent1_om_of_th2.tx)
            trainstate_val2_copy = TrainState.create(
                apply_fn=agent1_om_of_val2.apply_fn,
                params=agent1_om_of_val2.params,
                tx=agent1_om_of_val2.tx)

        # val update after loop no longer seems necessary

        key, subkey = jax.random.split(key)

        stuff = (subkey, trainstate_th1_copy, trainstate_val1_copy,
                 trainstate_th2_copy, trainstate_val2_copy,
                 trainstate_th1_ref, trainstate_val1_ref)

        stuff, aux = jax.lax.scan(one_outer_step_update_selfagent1, stuff, None, args.outer_steps)
        _, trainstate_th1_copy, trainstate_val1_copy, _, _, _, _ = stuff

        # Doing this just as a safety failcase scenario, and copy this at the end
        trainstate_after_outer_steps_th1 = TrainState.create(
            apply_fn=trainstate_th1_copy.apply_fn,
            params=trainstate_th1_copy.params,
            tx=trainstate_th1_copy.tx)
        trainstate_after_outer_steps_val1 = TrainState.create(
            apply_fn=trainstate_val1_copy.apply_fn,
            params=trainstate_val1_copy.params,
            tx=trainstate_val1_copy.tx)

        # --- START OF AGENT 2 UPDATE ---

        # Doing this just as a safety failcase scenario, to make sure each agent loop starts from the beginning
        trainstate_th1_copy = TrainState.create(
            apply_fn=trainstate_th1.apply_fn,
            params=trainstate_th1.params,
            tx=trainstate_th1.tx)
        trainstate_val1_copy = TrainState.create(
            apply_fn=trainstate_val1.apply_fn,
            params=trainstate_val1.params,
            tx=trainstate_val1.tx)
        trainstate_th2_copy = TrainState.create(
            apply_fn=trainstate_th2.apply_fn,
            params=trainstate_th2.params,
            tx=trainstate_th2.tx)
        trainstate_val2_copy = TrainState.create(
            apply_fn=trainstate_val2.apply_fn,
            params=trainstate_val2.params,
            tx=trainstate_val2.tx)


        if args.opp_model:
            key, subkey = jax.random.split(key)
            agent2_om_of_th1, agent2_om_of_val1 = opp_model_selfagent2(subkey, trainstate_th1_copy, trainstate_val1_copy,
                                 trainstate_th2_copy, trainstate_val2_copy, agent2_om_of_th1, agent2_om_of_val1)
            # No need to overwrite the refs for agent 1 because those aren't used in the outer loop as we're using KL div for agent 2
            # The inner KL div is done in the inner loop which will automatically recreate/save the ref before each set of inner loop steps
            trainstate_th1_copy = TrainState.create(
                apply_fn=agent2_om_of_th1.apply_fn,
                params=agent2_om_of_th1.params,
                tx=agent2_om_of_th1.tx)
            trainstate_val1_copy = TrainState.create(
                apply_fn=agent2_om_of_val1.apply_fn,
                params=agent2_om_of_val1.params,
                tx=agent2_om_of_val1.tx)


        key, subkey = jax.random.split(key)

        stuff = (subkey, trainstate_th1_copy, trainstate_val1_copy,
                 trainstate_th2_copy, trainstate_val2_copy,
                 trainstate_th2_ref, trainstate_val2_ref)

        stuff, aux = jax.lax.scan(one_outer_step_update_selfagent2, stuff, None,
                                  args.outer_steps)
        _, _, _, trainstate_th2_copy, trainstate_val2_copy, _, _ = stuff

        trainstate_after_outer_steps_th2 = TrainState.create(
            apply_fn=trainstate_th2_copy.apply_fn,
            params=trainstate_th2_copy.params,
            tx=trainstate_th2_copy.tx)
        trainstate_after_outer_steps_val2 = TrainState.create(
            apply_fn=trainstate_val2_copy.apply_fn,
            params=trainstate_val2_copy.params,
            tx=trainstate_val2_copy.tx)


        # TODO ensure this is correct. Ensure that the copy is updated on the outer loop once that has finished.
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

        vs_fixed_strats_score_record[0].append(score1rec)
        vs_fixed_strats_score_record[1].append(score2rec)

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

            if args.env == 'ipd':
                if args.inspect_ipd:
                    inspect_ipd(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2)

        if (update + 1) % args.checkpoint_every == 0:
            now = datetime.datetime.now()


            checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                        target=(trainstate_th1, trainstate_val1,
                                                trainstate_th2, trainstate_val2,
                                                coins_collected_info,
                                                score_record,
                                                vs_fixed_strats_score_record),
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

    np.random.seed(args.seed)



    if args.env == 'coin':
        assert args.grid_size == 3  # rest not implemented yet
        input_size = args.grid_size ** 2 * 4
        action_size = 4
        env = CoinGame()
    elif args.env == 'ipd':
        input_size = 6 # 3 * n_agents
        action_size = 2
        env = IPD(init_state_coop=args.init_state_coop, contrib_factor=args.contrib_factor)
    else:
        raise NotImplementedError("unknown env")
    vec_env_reset = jax.vmap(env.reset)
    vec_env_step = jax.vmap(env.step)



    key = jax.random.PRNGKey(args.seed)


    trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2 = get_init_trainstates(key, action_size, input_size)


    if args.load_dir is not None:
        epoch_num = int(args.load_prefix.split("epoch")[-1])
        if epoch_num % 10 == 0:
            epoch_num += 1  # Kind of an ugly temporary fix to allow for the updated checkpointing system which now has
            # record of rewards/eval vs fixed strat before the first training - important for IPD plots. Should really be applied to
            # all checkpoints with the new updated code I have, but the coin checkpoints above are from old code

        score_record = [jnp.zeros((2,))] * epoch_num
        vs_fixed_strats_score_record = [[jnp.zeros((3,))] * epoch_num,
                                        [jnp.zeros((3,))] * epoch_num]
        if args.env == 'coin':
            same_colour_coins_record = [jnp.zeros((1,))] * epoch_num
            diff_colour_coins_record = [jnp.zeros((1,))] * epoch_num
        else:
            same_colour_coins_record = []
            diff_colour_coins_record = []
        coins_collected_info = (
            same_colour_coins_record, diff_colour_coins_record)

        assert args.load_prefix is not None
        restored_tuple = checkpoints.restore_checkpoint(ckpt_dir=args.load_dir,
                                                        target=(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
                                                                coins_collected_info,
                                                                score_record,
                                                                vs_fixed_strats_score_record),
                                                        prefix=args.load_prefix)

        trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2, coins_collected_info, score_record, vs_fixed_strats_score_record = restored_tuple


    use_baseline = True
    if args.no_baseline:
        use_baseline = False

    assert args.inner_steps >= 1
    # Use 0 lr if you want no inner steps... TODO allow for 0 inner steps? Might save computation for naive learning instead of 0 lr
    assert args.outer_steps >= 1


    joint_scores = play(key, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
                        args.opp_model)
