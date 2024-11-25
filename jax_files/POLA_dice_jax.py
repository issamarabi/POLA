import numpy as np
import argparse
import datetime
import itertools

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


################################################################################
#                          Core DiCE / GAE Objectives                          #
################################################################################

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


################################################################################
#                           Acting in the Environment                          #
################################################################################

@jit
def act(scan_carry, _):
    """
    N-agent version: loop over each agent to get its action, log prob, and (optionally) value.

    scan_carry = (key, obs_batch, trainstates_p, trainstates_v, hidden_ps, hidden_vs)
      - key: PRNG key
      - obs_batch: [batch_size, obs_dim], the environment observations (shared or global)
      - trainstates_p: list of length n_agents with policy TrainStates
      - trainstates_v: list of length n_agents with value TrainStates (if use_baseline)
      - hidden_ps: list of length n_agents, each shape [batch_size, hidden_size] (policy RNN)
      - hidden_vs: list of length n_agents, same shape if use_baseline else None

    Returns:
      new_scan_carry, auxiliary
        - new_scan_carry is the updated state
        - auxiliary = (all_actions, all_log_probs, all_values, hidden_ps, hidden_vs,
                       all_softmax_logits, all_logits)
    """
    (key, obs_batch, p_states, v_states, hidden_p, hidden_v) = scan_carry
    n_agents = len(p_states)

    all_actions = []
    all_log_probs = []
    all_values = []
    all_logits = []
    all_softmax_probs = []

    # We will update hidden_ps[i], hidden_vs[i] for each agent i
    for i in range(n_agents):
        # Forward pass for the i-th agent's policy RNN
        hidden_p_i, logits_i = p_states[i].apply_fn(
            p_states[i].params,
            obs_batch,
            hidden_p[i]   # agent i's hidden state
        )
        dist_i = tfd.Categorical(logits=logits_i)
        key, subkey = jax.random.split(key)
        action_i = dist_i.sample(seed=subkey)
        logprob_i = dist_i.log_prob(action_i)

        # Forward pass value RNN if baselines are on
        if use_baseline:
            hidden_v_i, value_i = v_states[i].apply_fn(
                v_states[i].params,
                obs_batch,
                hidden_v[i]
            )
            value_i = value_i.squeeze(-1)  # shape [batch_size]
        else:
            hidden_v_i, value_i = None, jnp.zeros_like(logprob_i)

        # Update the hidden states in the lists
        hidden_p = hidden_p.at[i].set(hidden_p_i)
        if use_baseline:
            hidden_v = hidden_v.at[i].set(hidden_v_i)

        # Collect all agent i's data
        all_actions.append(action_i)
        all_log_probs.append(logprob_i)
        all_values.append(value_i)
        all_softmax_probs.append(jax.nn.softmax(logits_i))
        all_logits.append(logits_i)

    # Each list has length n_agents, each element shape [batch_size]
    # Stack them along axis=1 => shape [batch_size, n_agents]
    actions_arr = jnp.stack(all_actions, axis=1)
    log_probs_arr = jnp.stack(all_log_probs, axis=1)
    values_arr = jnp.stack(all_values, axis=1)
    # For softmax/logits we might want shape [batch_size, n_agents, action_dim]
    softmax_arr = jnp.stack(all_softmax_probs, axis=1)
    logits_arr = jnp.stack(all_logits, axis=1)

    new_scan_carry = (key, obs_batch, p_states, v_states, hidden_p, hidden_v)
    auxiliary = (actions_arr, log_probs_arr, values_arr,
                 hidden_p, hidden_v, softmax_arr, logits_arr)
    return new_scan_carry, auxiliary

@jit
def act_w_iter_over_obs(scan_carry, env_batch_obs):
    """
    Iterates over multiple time steps (env_batch_obs) via jax.lax.scan 
    and calls `act` for each time step.

    env_batch_obs: [batch_size, obs_dim] is the observation at this time step
                   that we pass to all agents.
    """
    (key, trainstates_p, trainstates_v, hidden_ps, hidden_vs) = scan_carry

    # We pass the same obs_batch to `act`, which loops over n_agents
    # and returns a stacked set of actions, log_probs, etc.
    act_input = (key, env_batch_obs, trainstates_p, trainstates_v, hidden_ps, hidden_vs)
    new_act_input, act_aux = act(act_input, None)

    (new_key, _, new_trainstates_p, new_trainstates_v,
     new_hidden_ps, new_hidden_vs) = new_act_input
    # act_aux is (actions, log_probs, values, hidden_ps, hidden_vs, softmax_probs, logits)

    new_scan_carry = (new_key, new_trainstates_p, new_trainstates_v,
                      new_hidden_ps, new_hidden_vs)
    return new_scan_carry, act_aux

@jit
def env_step(scan_carry, _):
    """
    Single environment step for n agents.
    Each agent acts using its policy RNN (all in parallel).
    Then the environment is stepped once.
    """
    (key, env_state, obs_batch,
     trainstates_p, trainstates_v,
     hidden_p, hidden_v) = scan_carry

    # 1) Sample actions for all agents.
    # ---------------------------------
    key, subkey = jax.random.split(key)
    scan_carry_act = (subkey, obs_batch, trainstates_p, trainstates_v, hidden_p, hidden_v)
    scan_carry_act, aux_act = act(scan_carry_act, None)
    (actions_arr, log_probs_arr, values_arr,
     hidden_p, hidden_v, softmax_arr, logits_arr) = aux_act

    # actions_arr:    [batch_size, n_agents]
    # log_probs_arr:  [batch_size, n_agents]
    # values_arr:     [batch_size, n_agents]
    # softmax_arr:    [batch_size, n_agents, action_dim]
    # logits_arr:     [batch_size, n_agents, action_dim]

    # 2) Step environment
    # --------------------
    env_subkeys = jax.random.split(key, args.batch_size)
    # Each row in actions_arr is the multi-agent action for that environment instance
    env_state_next, obs_next, rewards, env_info = vec_env_step(env_state, actions_arr, env_subkeys)
    # - env_state_next: updated environment states
    # - obs_next: next observation(s), shape [batch_size, obs_dim]
    # - rewards: shape [batch_size, n_agents]
    # - env_info: auxiliary info from the environment

    # 3) Build per-agent "aux" outputs
    # --------------------------------
    # Just like the 2-agent code built aux1, aux2, we can store
    # a list or tuple of (cat_probs, obs, log_prob, value, reward, action, etc.) for each agent i.
    # For convenience we can store everything in one big tuple, or an array of shape [n_agents, ...].
    # Below we keep them stacked so that lax.scan returns an array we can later index for each agent.

    # We'll store  for each agent i:
    #   - softmax_arr[:, i, :]   (policy distribution)
    #   - obs_next               (the next state, shared for all agents in fully observed tasks)
    #   - log_probs_arr[:, i]
    #   - values_arr[:, i]
    #   - rewards[:, i]
    #   - actions_arr[:, i]
    #   - We also could store actions of other agents, if needed.

    all_agents_aux = (
        softmax_arr,          # shape [batch_size, n_agents, action_dim]
        obs_next,             # shape [batch_size, obs_dim]
        log_probs_arr,        # shape [batch_size, n_agents]
        values_arr,           # shape [batch_size, n_agents]
        rewards,              # shape [batch_size, n_agents]
        actions_arr,          # shape [batch_size, n_agents]
        logits_arr,           # shape [batch_size, n_agents, action_dim]
        env_info              # any extra info from the environment
    )

    # 4) Update the scan_carry
    # -------------------------
    scan_carry_next = (
        key, env_state_next, obs_next,
        trainstates_p, trainstates_v,
        hidden_p, hidden_v
    )
    return scan_carry_next, all_agents_aux


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
def get_policies_for_states(
    key,
    trainstates_p,    # list of policy TrainStates for n_agents
    trainstates_v,    # list of value TrainStates for n_agents
    obs_hist          # shape [rollout_len, batch_size, obs_dim]
):
    """
    N-agent version: For a sequence of T (rollout_len) observations, returns the softmax action
    probabilities for each agent at each time step. Handles both single-batch
    (batch_size=1) and multi-batch scenarios.

    Returns: cat_act_probs_list with shape [T, batch_size, n_agents, action_size]
    """
    n_agents = len(trainstates_p)
    T = obs_hist.shape[0]  # rollout_len

    # Initialize hidden states for each agent: shape [batch_size, hidden_size]
    hidden_ps = [jnp.zeros((args.batch_size, args.hidden_size)) for _ in range(n_agents)]
    hidden_vs = [jnp.zeros((args.batch_size, args.hidden_size)) for _ in range(n_agents)] if use_baseline else [None]*n_agents

    key, subkey = jax.random.split(key)
    init_scan_carry = (subkey, trainstates_p, trainstates_v, hidden_ps, hidden_vs)

    def scan_body(scan_carry, obs_t):
        new_scan_carry, act_aux = act_w_iter_over_obs(scan_carry, obs_t)
        # act_aux => (actions, log_probs, values, hidden_ps, hidden_vs, softmax, logits)
        softmax_ = act_aux[5]  # shape [batch_size, n_agents, action_size]
        return new_scan_carry, softmax_

    final_scan_carry, cat_act_probs_seq = jax.lax.scan(
        scan_body, init_scan_carry, obs_hist, length=T
    )
    # cat_act_probs_seq => shape [T, batch_size, n_agents, action_size]
    return cat_act_probs_seq

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
    Returns list of hidden states for each agent's policy (and value) RNN
    """
    hidden_p = []
    hidden_v = []
    for _ in range(args.n_agents):
        hidden_p.append(jnp.zeros((args.batch_size, args.hidden_size)))
        if use_baseline:
            hidden_v.append(jnp.zeros((args.batch_size, args.hidden_size)))
        else:
            hidden_v.append(None)
    return hidden_p, hidden_v


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


###############################################################################
#    Outer-Loop Minimization for the "Self" Agent (POLA Outer Step)           #
###############################################################################

@partial(jit, static_argnums=(11))
def out_lookahead(key, th1, th1_params, val1, val1_params,
                  th2, th2_params, val2, val2_params,
                  old_th, old_val, self_agent=1):
    """
    The "outer" objective for self_agent. We do a rollout, sum the DiCE objective, 
    add a KL penalty wrt. old parameters (POLA). 
    """
    final_carry, aux, partial_state_hist = do_env_rollout(
        key, th1, th1_params, val1, val1_params,
        th2, th2_params, val2, val2_params, agent_for_state_history=self_agent)

    aux1, aux2, aux_info = aux
    (key_final, env_state, obs1, obs2,
     _, _, _, _, _, _, _, _, h_p1, h_v1, h_p2, h_v2) = final_carry

    if self_agent == 1:
        cat_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1
        partial_state_hist.extend(obs1_list)
        # final step val
        key_final, subkey = jax.random.split(key_final, 2)
        act_args1 = (subkey, obs1, th1, th1_params, val1, val1_params, h_p1, h_v1)
        _, (a1_end, lp1_end, v1_end, _, _, _, _) = act(act_args1, None)
        end_state_v = v1_end
        objective = dice_objective_plus_value_loss(lp1_list, lp2_list, r1_list, v1_list, end_state_v)
        # Collect probabilities, old probabilities
        key_final, subkey = jax.random.split(key_final, 2)
        self_pol_probs = get_policies_for_states(subkey, th1, th1_params, val1, val1_params, partial_state_hist)
        self_pol_probs_old = get_policies_for_states(subkey, old_th, old_th.params,
                                                     old_val, old_val.params, partial_state_hist)
    else:
        cat_probs2_list, obs2_list, lp2_list, lp1_list, v2_list, r2_list, a2_list, a1_list = aux2
        partial_state_hist.extend(obs2_list)
        key_final, subkey = jax.random.split(key_final, 2)
        act_args2 = (subkey, obs2, th2, th2_params, val2, val2_params, h_p2, h_v2)
        _, (a2_end, lp2_end, v2_end, _, _, _, _) = act(act_args2, None)
        end_state_v = v2_end
        objective = dice_objective_plus_value_loss(lp2_list, lp1_list, r2_list, v2_list, end_state_v)
        key_final, subkey = jax.random.split(key_final, 2)
        self_pol_probs = get_policies_for_states(subkey, th2, th2_params, val2, val2_params, partial_state_hist)
        self_pol_probs_old = get_policies_for_states(subkey, old_th, old_th.params,
                                                     old_val, old_val.params, partial_state_hist)

    if args.rev_kl:
        kl_div_term = rev_kl_div_jax(self_pol_probs, self_pol_probs_old)
    else:
        kl_div_term = kl_div_jax(self_pol_probs, self_pol_probs_old)

    return objective + args.outer_beta * kl_div_term, partial_state_hist

@jit
def one_outer_step_objective_selfagent1(
    key, th1_copy, th1_copy_params, val1_copy, val1_copy_params,
    th2_copy, th2_copy_params, val2_copy, val2_copy_params,
    th_ref, val_ref
):
    """
    Single outer-step objective for agent 1: 
    (1) run the inner step for agent 2, 
    (2) evaluate final objective for agent 1's vantage point.
    """
    # 1) Inner step: agent 2 updates
    key, subkey = jax.random.split(key)
    th2_after_inner, val2_after_inner = inner_steps_plus_update_otheragent2(
        subkey, th1_copy, th1_copy_params, val1_copy, val1_copy_params,
        th2_copy, th2_copy_params, val2_copy, val2_copy_params,
        th2_copy, val2_copy
    )
    # 2) Evaluate agent 1's objective
    if use_baseline:
        objective, state_hist = out_lookahead(
            key, th1_copy, th1_copy_params, val1_copy, val1_copy_params,
            th2_after_inner, th2_after_inner.params,
            val2_after_inner, val2_after_inner.params,
            th_ref, val_ref, self_agent=1
        )
    else:
        objective, state_hist = out_lookahead(
            key, th1_copy, th1_copy_params, None, None,
            th2_after_inner, th2_after_inner.params, None, None,
            th_ref, val_ref, self_agent=1
        )
    return objective, state_hist

@jit
def one_outer_step_objective_selfagent2(
    key, th1_copy, th1_copy_params, val1_copy, val1_copy_params,
    th2_copy, th2_copy_params, val2_copy, val2_copy_params,
    th_ref, val_ref
):
    """
    Single outer-step objective for agent 2: 
    (1) run the inner step for agent 1,
    (2) evaluate final objective for agent 2's vantage point.
    """
    # 1) Inner step: agent 1 updates
    key, subkey = jax.random.split(key)
    th1_after_inner, val1_after_inner = inner_steps_plus_update_otheragent1(
        subkey, th1_copy, th1_copy_params, val1_copy, val1_copy_params,
        th2_copy, th2_copy_params, val2_copy, val2_copy_params,
        th1_copy, val1_copy
    )
    # 2) Evaluate agent 2's objective
    if use_baseline:
        objective, state_hist = out_lookahead(
            key, th1_after_inner, th1_after_inner.params,
            val1_after_inner, val1_after_inner.params,
            th2_copy, th2_copy_params, val2_copy, val2_copy_params,
            th_ref, val_ref, self_agent=2
        )
    else:
        objective, state_hist = out_lookahead(
            key, th1_after_inner, th1_after_inner.params, None, None,
            th2_copy, th2_copy_params, None, None,
            th_ref, val_ref, self_agent=2
        )
    return objective, state_hist


###############################################################################
#                  JAX Scan Over Outer Steps (POLA Update)                    #
###############################################################################

@jit
def one_outer_step_update_selfagent1(scan_carry, _):
    """
    For agent 1's outer step update: 
    1) compute gradient of the objective wrt. agent 1's params,
    2) apply the gradient step,
    3) return updated agent 1 trainstates.
    """
    (key, th1_copy, val1_copy, th2_copy, val2_copy, th_ref, val_ref) = scan_carry
    key, subkey = jax.random.split(key)

    grad_fn = jax.grad(one_outer_step_objective_selfagent1, argnums=[2, 4], has_aux=True)
    (grad_th, grad_val), state_hist = grad_fn(
        subkey, th1_copy, th1_copy.params, val1_copy, val1_copy.params,
        th2_copy, th2_copy.params, val2_copy, val2_copy.params, th_ref, val_ref
    )
    th1_copy_updated = th1_copy.apply_gradients(grads=grad_th)
    val1_copy_updated = val1_copy.apply_gradients(grads=grad_val) if use_baseline else val1_copy

    # Only final trainstate returned; no need for aux data.
    updated_scan_carry = (
        key, th1_copy_updated, val1_copy_updated, th2_copy, val2_copy, th_ref, val_ref
    )
    return updated_scan_carry, state_hist

@jit
def one_outer_step_update_selfagent2(scan_carry, _):
    """
    For agent 2's outer step update. 
    1) compute gradient wrt. agent 2's params,
    2) apply step,
    3) return updated trainstates for agent 2.
    """
    (key, th1_copy, val1_copy, th2_copy, val2_copy, th_ref, val_ref) = scan_carry
    key, subkey = jax.random.split(key)

    grad_fn = jax.grad(one_outer_step_objective_selfagent2, argnums=[6, 8], has_aux=True)
    (grad_th, grad_val), state_hist = grad_fn(
        subkey, th1_copy, th1_copy.params, val1_copy, val1_copy.params,
        th2_copy, th2_copy.params, val2_copy, val2_copy.params, th_ref, val_ref
    )
    th2_copy_updated = th2_copy.apply_gradients(grads=grad_th)
    val2_copy_updated = val2_copy.apply_gradients(grads=grad_val) if use_baseline else val2_copy

    updated_scan_carry = (
        key, th1_copy, val1_copy, th2_copy_updated, val2_copy_updated, th_ref, val_ref
    )
    return updated_scan_carry, state_hist


###############################################################################
#          Evaluation vs. Fixed Strategies (ALLD, ALLC, TFT) & Helpers        #
###############################################################################

@jit
def eval_vs_alld(scan_carry, _):
    """
    Single-step logic for evaluating one agent (self_agent) while all others do 'ALLD'.
    We override the other agents' actions with 'defect' (IPD) or 'shortest path' (CoinGame).
    """
    (key, p_states, v_states,
     env_state, obs_batch,
     hidden_p, hidden_v,
     self_agent_idx) = scan_carry

    key, subkey = jax.random.split(key)
    scan_carry_act = (subkey, obs_batch, p_states, v_states, hidden_p, hidden_v)
    scan_carry_act, aux_act = act(scan_carry_act, None)
    (actions_arr, log_probs_arr, values_arr,
     hidden_p, hidden_v, softmax_arr, logits_arr) = aux_act  # shape [batch_size, n_agents, ...]

    # Overwrite the *other* agents' actions with 'defect' or 'move_to_coin'
    if args.env == "ipd":
        # 'defect' = 0
        # Keep self_agent_idx as is, override others
        def overwrite_ipd(a, i):
            return jnp.where(i == self_agent_idx, a, 0)
        agent_idxs = jnp.arange(actions_arr.shape[1])
        actions_arr = jax.vmap(
            lambda row: jax.vmap(overwrite_ipd, in_axes=(0,0))(row, agent_idxs)
        )(actions_arr)
    else:
        # 'defect' in coin => "move_towards_coin"
        env_subkeys = jax.random.split(key, args.batch_size)
        # get the "defect" action for each agent => shape [batch_size, n_agents]
        # then, for agent i != self_agent_idx, override
        moves_toward = env.get_moves_towards_coin(env_state)  # shape [batch_size, n_agents]
        def overwrite_coin(a, d, i):
            # keep a if i == self_agent_idx, else d
            return jnp.where(i == self_agent_idx, a, d)
        agent_idxs = jnp.arange(moves_toward.shape[1])
        actions_arr = jax.vmap(
            lambda row, def_row: jax.vmap(overwrite_coin, in_axes=(0,0,0))(row, def_row, agent_idxs)
        )(actions_arr, moves_toward)

    # Environment step
    env_subkeys = jax.random.split(key, args.batch_size)
    env_state_next, obs_next, rewards, env_info = vec_env_step(env_state, actions_arr, env_subkeys)

    # We'll measure the self-agent's average reward, but you can store all if you wish
    r_self = rewards[:, self_agent_idx].mean()

    # Build new scan_carry
    scan_carry_next = (
        key, p_states, v_states,
        env_state_next, obs_next,
        hidden_p, hidden_v,
        self_agent_idx
    )
    return scan_carry_next, r_self

@jit
def eval_vs_allc(scan_carry, _):
    """
    Evaluate how self_agent_idx does vs 'always cooperate'.
    """
    (key, p_states, v_states,
     env_state, obs_batch,
     hidden_p, hidden_v,
     self_agent_idx) = scan_carry

    key, subkey = jax.random.split(key)
    # 1) All n agents "act" ...
    scan_carry_act = (subkey, obs_batch, p_states, v_states, hidden_p, hidden_v)
    scan_carry_act, aux_act = act(scan_carry_act, None)
    (actions_arr, log_probs_arr, values_arr,
     hidden_p, hidden_v, softmax_arr, logits_arr) = aux_act

    # 2) Override the other n-1 agents with 'cooperate'
    if args.env == "ipd":
        # cooperate=1
        def overwrite_ipd(a, i):
            return jnp.where(i == self_agent_idx, a, 1)
        agent_idxs = jnp.arange(actions_arr.shape[1])
        actions_arr = jax.vmap(
            lambda row: jax.vmap(overwrite_ipd, in_axes=(0,0))(row, agent_idxs)
        )(actions_arr)
    else:
        # coin => "cooperative" is get_coop_actions
        coop_moves = env.get_coop_actions(env_state)  # shape [batch_size, n_agents]
        def overwrite_coin(a, c, i):
            return jnp.where(i == self_agent_idx, a, c)
        agent_idxs = jnp.arange(coop_moves.shape[1])
        actions_arr = jax.vmap(
            lambda row, coop_row: jax.vmap(overwrite_coin, in_axes=(0,0,0))(row, coop_row, agent_idxs)
        )(actions_arr, coop_moves)

    # 3) Environment step
    env_subkeys = jax.random.split(key, args.batch_size)
    env_state_next, obs_next, rewards, env_info = vec_env_step(env_state, actions_arr, env_subkeys)

    # 4) measure the self agent's reward
    r_self = rewards[:, self_agent_idx].mean()

    # 5) new carry
    scan_carry_next = (
        key, p_states, v_states,
        env_state_next, obs_next,
        hidden_p, hidden_v,
        self_agent_idx
    )
    return scan_carry_next, r_self

@jit
def eval_vs_tft(scan_carry, _):
    """
    Evaluate how self_agent_idx does vs n-1 agents playing "TFT".
    For IPD: each TFT agent i copies whatever i did last turn.
    For CoinGame: we need to define 'coin-based TFT' logic.
    """
    (key, p_states, v_states,
     env_state, obs_batch,
     hidden_p, hidden_v,
     self_agent_idx,
     prev_actions,  # shape [batch_size, n_agents]
     prev_coop_coin_flags,  # optional for coin
     r_prev          # if you want to track previous rewards
    ) = scan_carry

    # 1) Everyone "acts" normally
    key, subkey = jax.random.split(key)
    scan_carry_act = (subkey, obs_batch, p_states, v_states, hidden_p, hidden_v)
    scan_carry_act, aux_act = act(scan_carry_act, None)
    (actions_arr, log_probs_arr, values_arr,
     hidden_p, hidden_v, softmax_arr, logits_arr) = aux_act

    # 2) Overwrite n-1 "TFT" agents
    #    For IPD: they do: action[i] = prev_actions[i]
    #    For coin: if you used "flag=0 => defect, 1 => coop"
    #    or whatever logic you prefer
    if args.env == "ipd":
        def overwrite_ipd(a, i):
            # if i != self_agent_idx, do prev_actions[i]
            # else do a
            return jnp.where(i == self_agent_idx, a, prev_actions[:, i])
        # agent_idxs shape [n_agents]
        agent_idxs = jnp.arange(actions_arr.shape[1])
        # We must do a per-environment row override
        actions_arr = jax.vmap(
            lambda row_a, row_prev: jax.vmap(overwrite_ipd, in_axes=(None,0))(row_a, agent_idxs)
        )(actions_arr, prev_actions)
    else:
        # coin-based TFT => if you took my coin last turn, i 'defect' = get_moves_towards_coin
        # otherwise i 'cooperate' = get_coop_actions
        # This is purely an example - adapt to your actual logic
        # We'll just do the same approach as IPD for brevity or some custom approach
        # e.g. store a separate array that indicates "did you steal from me last time?"
        pass

    # 3) Step environment
    env_subkeys = jax.random.split(key, args.batch_size)
    env_state_next, obs_next, rewards, env_info = vec_env_step(env_state, actions_arr, env_subkeys)

    # 4) measure self agent's reward
    r_self = rewards[:, self_agent_idx].mean()

    # 5) Update carry with new prev_actions = actions_arr
    #    You might also update coin flags or store new r_prev
    scan_carry_next = (
        key, p_states, v_states,
        env_state_next, obs_next,
        hidden_p, hidden_v,
        self_agent_idx,
        actions_arr,      # the new prev_actions
        prev_coop_coin_flags,
        rewards          # or just r_self?
    )
    return scan_carry_next, r_self

@partial(jit, static_argnums=(4,))
def eval_vs_fixed_strategy(key, p_states, v_states, strat="alld", self_agent_idx=0):
    """
    Evaluate agent 'self_agent_idx' vs. a fixed strategy (alld, allc, tft).
    We run an episode of length 'args.rollout_len' in parallel across 'args.batch_size'.
    """
    # 1) Reset environment
    keys = jax.random.split(key, args.batch_size + 1)

    # Assumes consistent observation structure after reset.
    # If observation structures differ, consider padding or conditional handling in subsequent steps.
    env_state, obs_batch = vec_env_reset(keys[1:])

    # 2) init hidden states
    hidden_p = jnp.array([jnp.zeros((args.batch_size, args.hidden_size)) for _ in range(n_agents)])
    hidden_v = jnp.array([jnp.zeros((args.batch_size, args.hidden_size)) if use_baseline else None for _ in range(n_agents)], dtype=object)

    # 3) Build initial scan_carry
    # For TFT, you might need extra fields in the carry, e.g. prev_actions.
    # For alld/allc, we do not need them:
    scan_carry = (
        key, p_states, v_states,
        env_state, obs_batch,
        hidden_p, hidden_v,
        self_agent_idx
    )

    # 4) choose sub-function
    if strat == "alld":
        scan_func = eval_vs_alld
    elif strat == "allc":
        scan_func = eval_vs_allc
    elif strat == "tft":
        # If you want a big carry, define it properly or create a separate “tft_carry” version.
        raise NotImplementedError("eval_vs_tft not implemented.")
    else:
        raise ValueError("Unknown fixed strategy requested.")

    # 5) run the scan
    scan_carry, r_self_history = jax.lax.scan(scan_func, scan_carry,
                                              None, length=args.rollout_len)
    # r_self_history => shape [rollout_len,], each step's avg reward for self agent
    r_self_mean = r_self_history.mean()
    return r_self_mean


###############################################################################
#                   Opponent Modeling: Supervised Approach                    #
###############################################################################

@jit
def get_c_e_for_om(key, om_p, om_p_params, om_v, om_v_params,
                   other_state_history, other_act_history):
    """
    Opponent model's cross-entropy loss:
    - The "policy" is treated as a classifier to predict the actions of the other agent based on observed states.
    - The cross-entropy loss is computed using the other agent's actions (one-hot vectors) as target labels.
    - This effectively trains the opponent model's policy to mimic the other agent's behavior.

    Note:
    - The term `-p * log(p)` from the KL divergence drops out since `p` represents the target one-hot vectors (0 or 1).
    - Only the term `-p * log(q)` contributes to the gradient, aligning with standard cross-entropy computation.
    """
    key, subkey = jax.random.split(key)
    # Generate policy probabilities and hidden states for the other agent's observed states
    pol_probs, h_p_list, h_v_list = get_policies_and_h_for_states(subkey, om_p, om_p_params, om_v, om_v_params, other_state_history)

    # Cross-entropy loss: the policy's predicted probabilities (q) are evaluated against the target actions (one-hot encoded p)
    c_e_loss = -(other_act_history * jnp.log(pol_probs)).sum(axis=-1).mean()
    return c_e_loss, (h_p_list, h_v_list)

@jit
def get_val_loss_for_om(key, om_p, om_p_params, om_v, om_v_params,
                        other_state_history, other_act_history, rewards, end_state_v):
    pol_probs, pol_vals = get_policies_and_values_for_states(key, om_p, om_p_params, om_v, om_v_params, other_state_history)
    val_loss_ = value_loss(rewards, pol_vals, end_state_v)
    return val_loss_

def get_policies_and_h_for_states(key, train_p, train_p_params, train_v, train_v_params, obs_hist):
    """
    Variation of get_policies_for_states that also returns the hidden states at each step 
    (used for the OM's value function update).
    """
    h_p_init = jnp.zeros((args.batch_size, args.hidden_size))
    h_v_init = jnp.zeros((args.batch_size, args.hidden_size)) if use_baseline else None
    init_scan_carry = (key, train_p, train_p_params, train_v, train_v_params, h_p_init, h_v_init)
    obs_hist_for_scan = jnp.stack(obs_hist[:args.rollout_len], axis=0)
    final_scan_carry, aux_lists = jax.lax.scan(act_w_iter_over_obs, init_scan_carry, obs_hist_for_scan, args.rollout_len)
    (_, _, _, _, _, _, _), (a_list, lp_list, v_list, h_p_list, h_v_list, cat_probs_list, logits_list) = (final_scan_carry, aux_lists)
    return cat_probs_list, h_p_list, h_v_list

@jit
def opp_model_selfagent1_single_batch(scan_carry, _):
    """
    One "batch" of environment interactions for agent1 + OM training.
    """
    (key, train_th1, train_val1, true_th2, true_val2,
     om_th2, om_val2) = scan_carry
    key, subkey = jax.random.split(key)

    final_carry, aux_data, state_hist = do_env_rollout(
        subkey, train_th1, train_th1.params, train_val1, train_val1.params,
        true_th2, true_th2.params, true_val2, true_val2.params,
        agent_for_state_history=2
    )
    _, (aux2_cat_probs, obs2_list, lp2_list, lp1_list, v2_list, r2_list,
        a2_list, a1_list), _ = aux_data
    state_hist.extend(obs2_list)
    other_state_history = state_hist
    other_act_history = a2_list
    other_act_history_1hot = jax.nn.one_hot(other_act_history, action_size)

    # Cross-entropy grad
    om_grad_fn = jax.grad(get_c_e_for_om, argnums=2, has_aux=True)
    # Value grad
    om_val_grad_fn = jax.grad(get_val_loss_for_om, argnums=4) if use_baseline else None

    for _ in range(args.opp_model_steps_per_batch):
        key, subkey_ce = jax.random.split(key)
        grad_th2, h_lists = om_grad_fn(
            subkey_ce, om_th2, om_th2.params, om_val2, om_val2.params,
            other_state_history, other_act_history_1hot
        )
        h_p_list, h_v_list = h_lists
        om_th2 = om_th2.apply_gradients(grads=grad_th2)
        if use_baseline:
            key, subkey_val = jax.random.split(key)
            act_args2 = (subkey_val,
                         final_carry[3],  # obs2
                         om_th2, om_th2.params,
                         om_val2, om_val2.params,
                         h_p_list[-1], h_v_list[-1])
            _, aux2_act = act(act_args2, None)
            _, _, v2_end, _, _, _, _ = aux2_act
            grad_v2 = om_val_grad_fn(subkey_val,
                                     om_th2, om_th2.params,
                                     om_val2, om_val2.params,
                                     other_state_history,
                                     other_act_history_1hot,
                                     r2_list, v2_end)
            om_val2 = om_val2.apply_gradients(grads=grad_v2)

    return (key, train_th1, train_val1, true_th2, true_val2, om_th2, om_val2), None

@jit
def opp_model_selfagent2_single_batch(scan_carry, _):
    """
    One "batch" of environment interactions for agent2 + OM training.
    """
    (key, true_th1, true_val1, train_th2, train_val2,
     om_th1, om_val1) = scan_carry
    key, subkey = jax.random.split(key)

    final_carry, aux_data, state_hist = do_env_rollout(
        subkey, true_th1, true_th1.params, true_val1, true_val1.params,
        train_th2, train_th2.params, train_val2, train_val2.params,
        agent_for_state_history=1
    )
    (aux1_cat_probs, obs1_list, lp1_list, lp2_list, v1_list, r1_list,
     a1_list, a2_list), _, _ = aux_data
    state_hist.extend(obs1_list)
    other_state_history = state_hist
    other_act_history = a1_list
    other_act_history_1hot = jax.nn.one_hot(other_act_history, action_size)

    om_grad_fn = jax.grad(get_c_e_for_om, argnums=2, has_aux=True)
    om_val_grad_fn = jax.grad(get_val_loss_for_om, argnums=4) if use_baseline else None

    for _ in range(args.opp_model_steps_per_batch):
        key, subkey_ce = jax.random.split(key)
        grad_th1, h_lists = om_grad_fn(
            subkey_ce, om_th1, om_th1.params, om_val1, om_val1.params,
            other_state_history, other_act_history_1hot
        )
        h_p_list, h_v_list = h_lists
        om_th1 = om_th1.apply_gradients(grads=grad_th1)

        if use_baseline:
            key, subkey_val = jax.random.split(key)
            act_args1 = (subkey_val,
                         final_carry[2],  # obs1
                         om_th1, om_th1.params,
                         om_val1, om_val1.params,
                         h_p_list[-1], h_v_list[-1])
            _, aux1_act = act(act_args1, None)
            _, _, v1_end, _, _, _, _ = aux1_act
            grad_v1 = om_val_grad_fn(
                subkey_val, om_th1, om_th1.params, om_val1, om_val1.params,
                other_state_history, other_act_history_1hot, r1_list, v1_end
            )
            om_val1 = om_val1.apply_gradients(grads=grad_v1)

    return (key, true_th1, true_val1, train_th2, train_val2, om_th1, om_val1), None

@jit
def opp_model_selfagent1(key, th1, val1, true_th2, true_val2,
                         prev_om_th2, prev_om_val2):
    """
    Updates agent1's opponent model for agent2 by collecting environment data 
    without white-box access to agent2's parameters. Specifically:
      1. We gather trajectories of agent1 interacting with agent2 (using agent2's 
         'true' train states).
      2. We train an opponent model (om_th2, om_val2) to mimic agent2's actions 
         and optionally its value function by supervised learning.

    This is achieved by:
      - Creating new TrainState copies for om_th2, om_val2 from prev_om_th2, prev_om_val2.
      - Repeatedly scanning over environment rollouts (opp_model_selfagent1_single_batch).
      - Returning the updated opponent model train states.

    Args:
        key: JAX PRNGKey for randomness.
        th1, val1: Agent1's policy and value TrainStates.
        true_th2, true_val2: The 'true' policy and value TrainStates for agent2, 
                             used only for generating trajectories (not read directly).
        prev_om_th2, prev_om_val2: The previously learned opponent model TrainStates 
                                   for agent2's policy/value.

    Returns:
        (om_th2, om_val2): Updated opponent model TrainStates for agent2.
    """

    om_th2 = TrainState.create(apply_fn=prev_om_th2.apply_fn,
                               params=prev_om_th2.params,
                               tx=prev_om_th2.tx)
    om_val2 = TrainState.create(apply_fn=prev_om_val2.apply_fn,
                                params=prev_om_val2.params,
                                tx=prev_om_val2.tx)
    scan_carry = (key, th1, val1, true_th2, true_val2, om_th2, om_val2)
    scan_carry, _ = jax.lax.scan(
        opp_model_selfagent1_single_batch, scan_carry, None, args.opp_model_data_batches
    )
    return scan_carry[5], scan_carry[6]  # om_th2, om_val2

@jit
def opp_model_selfagent2(key, true_th1, true_val1, th2, val2,
                         prev_om_th1, prev_om_val1):
    """
    For agent2's perspective: gather environment data with (true_th1, true_val1) vs (th2, val2),
    then train an opponent model (om_th1, om_val1).
    """
    om_th1 = TrainState.create(apply_fn=prev_om_th1.apply_fn,
                               params=prev_om_th1.params,
                               tx=prev_om_th1.tx)
    om_val1 = TrainState.create(apply_fn=prev_om_val1.apply_fn,
                                params=prev_om_val1.params,
                                tx=prev_om_val1.tx)
    scan_carry = (key, true_th1, true_val1, th2, val2, om_th1, om_val1)
    scan_carry, _ = jax.lax.scan(
        opp_model_selfagent2_single_batch, scan_carry, None, args.opp_model_data_batches
    )
    return scan_carry[5], scan_carry[6]  # om_th1, om_val1


###############################################################################
#                           Main Training Loop                                #
###############################################################################


def get_init_trainstates(key, n_agents, action_size_, input_size_):
    """
    Create a policy and value TrainState for EACH of the n_agents.
    Return them as two lists:
        trainstate_th[i] = policy TrainState for agent i
        trainstate_val[i] = value TrainState for agent i
    """
    # We create separate RNN modules for each agent
    # and separate train states for each agent's policy and value.

    n_keys_needed = 1 + 2*n_agents
    agent_keys = jax.random.split(key, n_keys_needed)[1:]

    # Prepare the chosen outer optimizer & value optimizer
    if args.optim.lower() == 'adam':
        theta_optimizer = optax.adam(learning_rate=args.lr_out)
        value_optimizer = optax.adam(learning_rate=args.lr_v)
    elif args.optim.lower() == 'sgd':
        theta_optimizer = optax.sgd(learning_rate=args.lr_out)
        value_optimizer = optax.sgd(learning_rate=args.lr_v)
    else:
        raise Exception("Unknown or Not Implemented Optimizer")

    trainstate_th = []
    trainstate_val = []

    for i in range(n_agents):
        key_p = agent_keys[2*i]
        key_v = agent_keys[2*i+1]

        # Create the RNN modules
        theta_p = RNN(num_outputs=action_size_,
                      num_hidden_units=args.hidden_size,
                      layers_before_gru=args.layers_before_gru)
        theta_v = RNN(num_outputs=1,
                      num_hidden_units=args.hidden_size,
                      layers_before_gru=args.layers_before_gru)

        # Initialize parameters
        theta_p_params = theta_p.init(key_p,
                                      jnp.ones([args.batch_size, input_size_]),
                                      jnp.zeros(args.hidden_size))
        theta_v_params = theta_v.init(key_v,
                                      jnp.ones([args.batch_size, input_size_]),
                                      jnp.zeros(args.hidden_size))

        # Create TrainStates
        train_p_state = TrainState.create(
            apply_fn=theta_p.apply,
            params=theta_p_params,
            tx=theta_optimizer
        )
        train_v_state = TrainState.create(
            apply_fn=theta_v.apply,
            params=theta_v_params,
            tx=value_optimizer
        )

        trainstate_th.append(train_p_state)
        trainstate_val.append(train_v_state)

    return trainstate_th, trainstate_val


@jit
def eval_progress(subkey, th1, val1, th2, val2):
    """
    Quick environment rollout (single pass) + evaluation vs fixed strategies 
    for diagnostics/logging.
    """
    keys = jax.random.split(subkey, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]
    env_state, obsv = vec_env_reset(env_subkeys)
    obs1 = obsv
    obs2 = obsv
    h_p1, h_p2, h_v1, h_v2 = get_init_hidden_states()

    init_scan_carry = (key, env_state, obs1, obs2,
                       th1, th1.params, val1, val1.params,
                       th2, th2.params, val2, val2.params,
                       h_p1, h_v1, h_p2, h_v2)
    final_scan_carry, (aux1, aux2, aux_info) = jax.lax.scan(env_step, init_scan_carry, None, args.rollout_len)

    (cat_probs1, obs_out1, lp1, lp2, v1, r1, a1, a2) = aux1
    (cat_probs2, obs_out2, lp2b, lp1b, v2, r2, a2b, a1b) = aux2
    avg_r1 = r1.mean()
    avg_r2 = r2.mean()

    rr_matches_amount = None
    rb_matches_amount = None
    br_matches_amount = None
    bb_matches_amount = None

    if args.env == 'coin':
        (rr_matches, rb_matches, br_matches, bb_matches) = aux_info
        rr_matches_amount = rr_matches.sum(axis=0).mean()
        rb_matches_amount = rb_matches.sum(axis=0).mean()
        br_matches_amount = br_matches.sum(axis=0).mean()
        bb_matches_amount = bb_matches.sum(axis=0).mean()

    # Evaluate vs fixed strats
    score1rec = []
    score2rec = []
    for strat in ["alld", "allc", "tft"]:
        key, subkey = jax.random.split(key)
        sc1, _ = eval_vs_fixed_strategy(subkey, th1, val1, strat, self_agent=1)
        score1rec.append(sc1[0])

        key, subkey = jax.random.split(key)
        sc2, _ = eval_vs_fixed_strategy(subkey, th2, val2, strat, self_agent=2)
        score2rec.append(sc2[1])

    return avg_r1, avg_r2, rr_matches_amount, rb_matches_amount, br_matches_amount, bb_matches_amount, jnp.stack(score1rec), jnp.stack(score2rec)


def inspect_ipd(trainstates_p, trainstates_val):
    """
    Inspect policies of TWO agents in IPD env across all possible state histories up to two steps.
    """

    # Initialize PRNG key and reset the vectorized environment
    key = jax.random.PRNGKey(0)
    unused_keys = jax.random.split(key, args.batch_size)
    init_state = env.init_state

    # Generate all possible combinations of states for both agents (2 actions each)
    state_combinations = list(itertools.product(range(2), repeat=2))  # [(0,0), (0,1), (1,0), (1,1)]
    all_state_histories = itertools.product(state_combinations, repeat=2)  # 16 combinations

    for (i, j), (ii, jj) in all_state_histories:
        state1 = env.states[i, j]
        state2 = env.states[ii, jj]
        obs_hist = jnp.array([init_state, state1, state2]).reshape(3, 1, -1)

        print(f"\nState History: {obs_hist}")

        pol_probs = get_policies_for_states(
            key, trainstates_p, trainstates_val, obs_hist
        )
        print(f"Policy Probabilities: {pol_probs}")


def play(key, init_th1, init_val1, init_th2, init_val2, use_opp_model=False):
    """
    Main training loop that repeatedly:
      1) Saves snapshot of performance,
      2) Runs the outer-step for agent 1,
      3) Runs the outer-step for agent 2,
      4) Logs progress & optionally writes checkpoints.
    """
    print("start iterations with", args.inner_steps, "inner steps and", args.outer_steps, "outer steps:")
    score_record = []
    vs_fixed_strats_score_record = [[], []]
    same_colour_coins_record = []
    diff_colour_coins_record = []

    th1 = TrainState.create(apply_fn=init_th1.apply_fn, params=init_th1.params, tx=init_th1.tx)
    val1 = TrainState.create(apply_fn=init_val1.apply_fn, params=init_val1.params, tx=init_val1.tx)
    th2 = TrainState.create(apply_fn=init_th2.apply_fn, params=init_th2.params, tx=init_th2.tx)
    val2 = TrainState.create(apply_fn=init_val2.apply_fn, params=init_val2.params, tx=init_val2.tx)

    if args.opp_model:
        key, subkey = jax.random.split(key)
        th, val = get_init_trainstates(subkey, 2, action_size, input_size)
        agent1_om_th2, agent1_om_val2, agent2_om_th1, agent2_om_val1 = th[0], val[0], th[1], val[1]


    # Evaluate initial performance
    key, subkey = jax.random.split(key)
    score1, score2, rr_matches, rb_matches, br_matches, bb_matches, sc1rec, sc2rec = eval_progress(subkey, th1, val1, th2, val2)
    if args.env == "coin":
        same_colour_coins_record.append(rr_matches + bb_matches)
        diff_colour_coins_record.append(rb_matches + br_matches)
    vs_fixed_strats_score_record[0].append(sc1rec)
    vs_fixed_strats_score_record[1].append(sc2rec)
    score_record.append(jnp.stack((score1, score2)))

    for update_idx in range(args.n_update):
        # For referencing the agent's old parameters in the KL penalty (NOT TO BE UPDATED)
        th1_ref = TrainState.create(apply_fn=th1.apply_fn, params=th1.params, tx=th1.tx)
        val1_ref = TrainState.create(apply_fn=val1.apply_fn, params=val1.params, tx=val1.tx)
        th2_ref = TrainState.create(apply_fn=th2.apply_fn, params=th2.params, tx=th2.tx)
        val2_ref = TrainState.create(apply_fn=val2.apply_fn, params=val2.params, tx=val2.tx)

        #----------------- Agent 1 Outer Update -----------------#
        th1_copy = TrainState.create(apply_fn=th1.apply_fn, params=th1.params, tx=th1.tx)
        val1_copy = TrainState.create(apply_fn=val1.apply_fn, params=val1.params, tx=val1.tx)
        th2_copy = TrainState.create(apply_fn=th2.apply_fn, params=th2.params, tx=th2.tx)
        val2_copy = TrainState.create(apply_fn=val2.apply_fn, params=val2.params, tx=val2.tx)

        if args.opp_model:
            key, subkey_om = jax.random.split(key)
            agent1_om_th2, agent1_om_val2 = opp_model_selfagent1(
                subkey_om, th1_copy, val1_copy, th2_copy, val2_copy,
                agent1_om_th2, agent1_om_val2
            )
            th2_copy = TrainState.create(apply_fn=agent1_om_th2.apply_fn, params=agent1_om_th2.params, tx=agent1_om_th2.tx)
            val2_copy = TrainState.create(apply_fn=agent1_om_val2.apply_fn, params=agent1_om_val2.params, tx=agent1_om_val2.tx)

        key, subkey_o1 = jax.random.split(key)
        init_scan_carry = (subkey_o1, th1_copy, val1_copy, th2_copy, val2_copy, th1_ref, val1_ref)
        final_scan_carry, _ = jax.lax.scan(one_outer_step_update_selfagent1, init_scan_carry, None, args.outer_steps)
        (_, th1_copy_updated, val1_copy_updated, _, _, _, _) = final_scan_carry

        #----------------- Agent 2 Outer Update -----------------#
        # Reset copies to the original main states ( so agent2 sees agent1's real final from prev iteration, etc.)
        th1_copy2 = TrainState.create(apply_fn=th1.apply_fn, params=th1.params, tx=th1.tx)
        val1_copy2 = TrainState.create(apply_fn=val1.apply_fn, params=val1.params, tx=val1.tx)
        th2_copy2 = TrainState.create(apply_fn=th2.apply_fn, params=th2.params, tx=th2.tx)
        val2_copy2 = TrainState.create(apply_fn=val2.apply_fn, params=val2.params, tx=val2.tx)

        if args.opp_model:
            key, subkey_om2 = jax.random.split(key)
            agent2_om_th1, agent2_om_val1 = opp_model_selfagent2(
                subkey_om2, th1_copy2, val1_copy2, th2_copy2, val2_copy2,
                agent2_om_th1, agent2_om_val1
            )
            th1_copy2 = TrainState.create(apply_fn=agent2_om_th1.apply_fn, params=agent2_om_th1.params, tx=agent2_om_th1.tx)
            val1_copy2 = TrainState.create(apply_fn=agent2_om_val1.apply_fn, params=agent2_om_val1.params, tx=agent2_om_val1.tx)

        key, subkey_o2 = jax.random.split(key)
        init_scan_carry2 = (subkey_o2, th1_copy2, val1_copy2, th2_copy2, val2_copy2, th2_ref, val2_ref)
        final_scan_carry2, _ = jax.lax.scan(one_outer_step_update_selfagent2, init_scan_carry2, None, args.outer_steps)
        (_, th1_copy2_, val1_copy2_, th2_copy_updated, val2_copy_updated, _, _) = final_scan_carry2

        # Overwrite main states with final updated copies
        th1 = th1_copy_updated
        val1 = val1_copy_updated
        th2 = th2_copy_updated
        val2 = val2_copy_updated

        # Evaluate progress:
        key, subkey_eval = jax.random.split(key)
        s1, s2, rr_m, rb_m, br_m, bb_m, s1rec, s2rec = eval_progress(subkey_eval, th1, val1, th2, val2)
        if args.env == 'coin':
            same_colour_coins_record.append(rr_m + bb_m)
            diff_colour_coins_record.append(rb_m + br_m)
        vs_fixed_strats_score_record[0].append(s1rec)
        vs_fixed_strats_score_record[1].append(s2rec)
        score_record.append(jnp.stack((s1, s2)))

        if (update_idx + 1) % args.print_every == 0:
            print("*" * 10)
            print(f"Epoch: {update_idx + 1}")
            print(f"Agent 1 Score: {s1}")
            print(f"Agent 2 Score: {s2}")
            if args.env == 'coin':
                print("Same-colour coin pickups:", rr_m + bb_m)
                print("Diff-colour coin pickups:", rb_m + br_m)
                print("RR coins:", rr_m)
                print("RB coins:", rb_m)
                print("BR coins:", br_m)
                print("BB coins:", bb_m)
            print("Scores vs [ALLD, ALLC, TFT] for Agent1:", s1rec)
            print("Scores vs [ALLD, ALLC, TFT] for Agent2:", s2rec)
                        
            if args.env == 'ipd' and args.inspect_ipd:
                inspect_ipd(th1, val1, th2, val2)

        if (update_idx + 1) % args.checkpoint_every == 0:
            now = datetime.datetime.now()
            checkpoints.save_checkpoint(
                ckpt_dir=args.save_dir,
                target=(
                    th1, val1, th2, val2,
                    (same_colour_coins_record, diff_colour_coins_record),
                    score_record,
                    vs_fixed_strats_score_record
                ),
                step=update_idx + 1,
                prefix=f"checkpoint_{now.strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch"
            )

    return []  # or return any final metrics


################################################################################
#                             Main Script Entry                                #
################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser("POLA")

    # -------------------- General Hyperparameters --------------------
    parser.add_argument("--inner_steps", type=int, default=1, help="Number of inner loop steps (K).")
    parser.add_argument("--outer_steps", type=int, default=1, help="Number of outer loop steps (M).")
    parser.add_argument("--lr_out", type=float, default=0.005, help="Outer loop learning rate.")
    parser.add_argument("--lr_in", type=float, default=0.03, help="Inner loop learning rate.")
    parser.add_argument("--lr_v", type=float, default=0.001, help="Learning rate for the value function.")
    parser.add_argument("--gamma", type=float, default=0.96, help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=1.0, help="GAE lambda parameter.")
    parser.add_argument("--n_update", type=int, default=5000, help="Number of main training epochs.")
    parser.add_argument("--rollout_len", type=int, default=50, help="Time horizon of game (episode length).")
    parser.add_argument("--batch_size", type=int, default=4000, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--hidden_size", type=int, default=32, help="Number of hidden units in RNN.")
    parser.add_argument("--layers_before_gru", type=int, default=2,
                        choices=[0, 1, 2],
                        help="Number of Dense layers before the GRU cell.")
    parser.add_argument("--optim", type=str, default="adam",
                        help="Optimizer for the outer loop (adam or sgd).")

    # -------------------- Environment / Task --------------------
    parser.add_argument("--env", type=str, default="coin", choices=["ipd", "coin"],
                        help="Which environment to run: 'ipd' or 'coin'.")
    parser.add_argument("--n_agents", type=int, default=2, help="Number of agents.")
    parser.add_argument("--grid_size", type=int, default=3,
                        help="Grid size for Coin Game (only 3 is implemented as of 2/2/2025).")
    parser.add_argument("--contrib_factor", type=float, default=1.33,
                        help="Contribution factor in IPD-like environment.")
    parser.add_argument("--init_state_coop", action="store_true",
                        help="If True, IPD starts at (C,C) instead of a special init state.")
    parser.add_argument("--diff_coin_reward", type=float, default=1.0,
                        help="Reward for picking up opponent's coin in Coin Game.")
    parser.add_argument("--diff_coin_cost", type=float, default=-2.0,
                        help="Cost inflicted on opponent when you pick up their coin.")
    parser.add_argument("--same_coin_reward", type=float, default=1.0,
                        help="Reward for picking up your own coin.")
    parser.add_argument("--split_coins", action="store_true",
                        help="If both agents collect simultaneously, split the reward. (not always implemented).")

    # -------------------- Proximal & Baseline Settings --------------------
    parser.add_argument("--outer_beta", type=float, default=0.0, help="Outer KL penalty coefficient (beta_out).")
    parser.add_argument("--inner_beta", type=float, default=0.0, help="Inner KL penalty coefficient (beta_in).")
    parser.add_argument("--rev_kl", action="store_true",
                        help="If True, use reverse KL. Otherwise, forward KL.")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Disable baseline/critic (GAE). Then uses basic DiCE.")
    parser.add_argument("--zero_vals", action="store_true",
                        help="For debugging: forcibly set all baseline values to zero in Loaded DiCE.")
    parser.add_argument("--val_update_after_loop", action="store_true",
                        help="Update value functions only after the outer POLA loop finishes")

    # -------------------- Opponent Modeling Settings --------------------
    parser.add_argument("--opp_model", action="store_true",
                        help="Enable opponent modeling. Agents do not directly see each other's parameters.")
    parser.add_argument("--opp_model_steps_per_batch", type=int, default=1,
                        help="Number of supervised gradient steps on each mini-batch for opponent modeling.")
    parser.add_argument("--opp_model_data_batches", type=int, default=100,
                        help="Number of environment rollout batches for data collection in OM step to train opp model.")
    parser.add_argument("--om_lr_p", type=float, default=0.005,
                        help="Learning rate for opponent model policy (BC).")
    parser.add_argument("--om_lr_v", type=float, default=0.001,
                        help="Learning rate for opponent model value function (BC).")

    # -------------------- Logging / Saving / Debug --------------------
    parser.add_argument("--print_every", type=int, default=1,
                        help="Print logs every 'print_every' epochs.")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save a checkpoint every 'checkpoint_every' epochs.")
    parser.add_argument("--save_dir", type=str, default='.',
                        help="Directory to save checkpoints.")
    parser.add_argument("--load_dir", type=str, default=None,
                        help="Directory from which to load checkpoints.")
    parser.add_argument("--load_prefix", type=str, default=None,
                        help="Checkpoint prefix for loading.")
    parser.add_argument("--inspect_ipd", action="store_true",
                        help="If set, prints detailed info in IPD states for debugging.")

    # -------------------- Additional Settings --------------------
    parser.add_argument("--hist_one", action="store_true",
                        help="Use one step history (no GRU or RNN, just one step history)")
    parser.add_argument("--print_info_each_outer_step", action="store_true",
                        help="Print information at each outer step for debugging purposes")
    parser.add_argument("--std", type=float, default=0.1,
                        help="Standard deviation for initialization of policy/value parameters")

    global args
    args = parser.parse_args()
    global use_baseline
    use_baseline = not args.no_baseline

    # Set random seed
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Setup the environment
    n_agents = args.n_agents

    if args.env == 'coin':
        env = CoinGame(n_agents=n_agents, grid_size=args.grid_size)
        input_size = 2 * n_agents * args.grid_size * args.grid_size
        action_size = 4  # up / down / left / right

    elif args.env == 'ipd':
        env = IPD(n_agents=n_agents,
                  start_with_cooperation=args.init_state_coop,
                  cooperation_factor=args.contrib_factor)
        input_size = 3 * n_agents
        action_size = 2  # cooperate or defect

    else:
        raise NotImplementedError("Unsupported environment type.")

    # Vectorize reset and step
    vec_env_reset = jax.vmap(env.reset)
    vec_env_step = jax.vmap(env.step)

    # Create *lists* of initial TrainState objects for each of the n_agents
    trainstates_p, trainstates_val = get_init_trainstates(key, n_agents, action_size, input_size)

    # Optionally load from a checkpoint
    if args.load_dir is not None and args.load_prefix is not None:
        # Extract epoch number from load_prefix
        try:
            epoch_num = int(args.load_prefix.split("epoch")[-1])
        except (IndexError, ValueError):
            raise ValueError("Failed to extract epoch number from load_prefix. Ensure it contains 'epoch<NUMBER>'.")

        # Apply the same adjustment as the original code
        if epoch_num % 10 == 0:
            epoch_num += 1  # Temporary fix as per original code comments

        # Initialize records based on number of agents
        score_record = [jnp.zeros((args.n_agents,))] * epoch_num
        vs_fixed_strats_score_record = [
            [jnp.zeros((3,))] * epoch_num for _ in range(args.n_agents)
        ]

        # Initialize coins_collected_info based on environment
        if args.env == 'coin':
            coins_collected_info = (
                [jnp.zeros((1,))] * epoch_num,
                [jnp.zeros((1,))] * epoch_num
            )
        else:
            coins_collected_info = ([], [])

        # Restore checkpoint
        restored_tuple = checkpoints.restore_checkpoint(
            ckpt_dir=args.load_dir,
            target=(
                *trainstates_p, *trainstates_val,
                coins_collected_info,
                score_record,
                vs_fixed_strats_score_record
            ),
            step=epoch_num,
            prefix=args.load_prefix
        )

        # Unpack the restored data
        # Assuming the order matches the initialization
        trainstates_p = list(restored_tuple[:args.n_agents])
        trainstates_val = list(restored_tuple[args.n_agents:2 * args.n_agents])
        coins_collected_info = restored_tuple[2 * args.n_agents]
        score_record = restored_tuple[2 * args.n_agents + 1]
        vs_fixed_strats_score_record = restored_tuple[2 * args.n_agents + 2]

    # Finally, run the main training loop
    play(
        key,
        trainstates_p, trainstates_val,  # List of policy and value TrainStates for all agents
        trainstates_p, trainstates_val,  # Pass the same lists as 'self' agents (to be handled in 'play')
        use_opp_model=args.opp_model
    )
