"""
Refactored result_plots.py

This script loads checkpoint data and plots evaluation results for either the coin
or the IPD environment.
"""

import jax.numpy as jnp
import jax.random
import numpy as np
from matplotlib import pyplot as plt

from POLA_dice_jax import RNN
from flax.training.train_state import TrainState
import optax
from flax.training import checkpoints

# =============================================================================
# Configuration Parameters
# =============================================================================

LOAD_DIR = "."
HIDDEN_SIZE = 64
BATCH_SIZE = 2000
LR_OUT = 0.005
LR_V = 0.00005
OUTER_OPTIM = "adam"  # "adam" or "sgd"
LAYERS_BEFORE_GRU = 1
PLOT_COIN = False  # Set to False to plot IPD results

if PLOT_COIN:
    ACTION_SIZE = 4
    INPUT_SIZE = 36
    # Checkpoint lists for coin environment (each list entry can be a string or a list of strings)
    CKPTS_POLA = [
        "checkpoint_2025-02-08_18-41_seed1_epoch100"
    ]
    CKPTS_LOLA = [

        "checkpoint_2022-10-04_06-23_seed9_epoch50000",
        "checkpoint_2022-10-04_06-14_seed10_epoch50000"
    ]
    CKPTS_POLA_OM = [
        "checkpoint_2022-09-20_09-31_seed1_epoch251",
        "checkpoint_2022-09-30_07-34_seed10_epoch251"
    ]
else:
    ACTION_SIZE = 2
    INPUT_SIZE = 6
    CKPTS_POLA = [
        "checkpoint_2022-10-06_12-09_seed9_epoch150",
        "checkpoint_2022-10-06_12-07_seed10_epoch150"
    ]
    CKPTS_LOLA = [
        "checkpoint_2022-10-05_08-21_seed1_epoch20000",
        "checkpoint_2022-10-05_18-50_seed10_epoch20000"
    ]
    CKPTS_POLA_OM = [
        "checkpoint_2022-10-06_13-27_seed9_epoch150",
        "checkpoint_2022-10-06_13-27_seed10_epoch150"
    ]
    CKPTS_STATIC = []  # Not used in IPD


# =============================================================================
# Loading Checkpoints and Computing Scores
# =============================================================================

def load_from_checkpoint(load_dir, load_prefix, action_size, hidden_size, batch_size,
                         input_size, lr_out, lr_v, outer_optim):
    """
    Load a checkpoint and initialize the training states and evaluation records.
    
    Returns:
        Tuple (score_record, vs_fixed_strats_score_record, coins_collected_info)
    """
    # Extract epoch number from the checkpoint name and adjust if necessary.
    epoch_num = int(load_prefix.split("epoch")[-1])

    if epoch_num % 10 == 0:
        epoch_num += 1  # Temporary fix for checkpointing system with record of eval vs fixed strat pre training

    # Initialize models for policy and value functions (for two agents)
    theta_p1 = RNN(num_outputs=action_size, num_hidden_units=hidden_size,
                   layers_before_gru=LAYERS_BEFORE_GRU)
    theta_v1 = RNN(num_outputs=1, num_hidden_units=hidden_size,
                   layers_before_gru=LAYERS_BEFORE_GRU)
    theta_p2 = RNN(num_outputs=action_size, num_hidden_units=hidden_size,
                   layers_before_gru=LAYERS_BEFORE_GRU)
    theta_v2 = RNN(num_outputs=1, num_hidden_units=hidden_size,
                   layers_before_gru=LAYERS_BEFORE_GRU)

    # Initialize parameters with dummy input.
    theta_p1_params = theta_p1.init(jax.random.PRNGKey(0), jnp.ones([batch_size, input_size]), jnp.zeros(hidden_size))
    theta_v1_params = theta_v1.init(jax.random.PRNGKey(0), jnp.ones([batch_size, input_size]), jnp.zeros(hidden_size))
    theta_p2_params = theta_p2.init(jax.random.PRNGKey(0), jnp.ones([batch_size, input_size]), jnp.zeros(hidden_size))
    theta_v2_params = theta_v2.init(jax.random.PRNGKey(0), jnp.ones([batch_size, input_size]), jnp.zeros(hidden_size))

    # Set up the outer and value optimizers.
    if outer_optim.lower() == 'adam':
        theta_optimizer = optax.adam(learning_rate=lr_out)
        value_optimizer = optax.adam(learning_rate=lr_v)
    elif outer_optim.lower() == 'sgd':
        theta_optimizer = optax.sgd(learning_rate=lr_out)
        value_optimizer = optax.sgd(learning_rate=lr_v)
    else:
        raise Exception("Unknown or Not Implemented Optimizer")

    # Create TrainState objects.
    trainstate_th1 = TrainState.create(apply_fn=theta_p1.apply, params=theta_p1_params, tx=theta_optimizer)
    trainstate_val1 = TrainState.create(apply_fn=theta_v1.apply, params=theta_v1_params, tx=value_optimizer)
    trainstate_th2 = TrainState.create(apply_fn=theta_p2.apply, params=theta_p2_params, tx=theta_optimizer)
    trainstate_val2 = TrainState.create(apply_fn=theta_v2.apply, params=theta_v2_params, tx=value_optimizer)

    score_record = [jnp.zeros((2,))] * epoch_num
    vs_fixed_strats_score_record = [[jnp.zeros((3,))] * epoch_num, [jnp.zeros((3,))] * epoch_num]
    if PLOT_COIN:
        same_colour_coins_record = [jnp.zeros((1,))] * epoch_num
        diff_colour_coins_record = [jnp.zeros((1,))] * epoch_num
    else:
        same_colour_coins_record, diff_colour_coins_record = [], []
    coins_collected_info = (same_colour_coins_record, diff_colour_coins_record)

    restored_tuple = checkpoints.restore_checkpoint(
        ckpt_dir=load_dir,
        target=(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
                coins_collected_info, score_record, vs_fixed_strats_score_record),
        prefix=load_prefix
    )

    (trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
     coins_collected_info, score_record, vs_fixed_strats_score_record) = restored_tuple

    if PLOT_COIN:
        same_colour_coins_record, diff_colour_coins_record = coins_collected_info
        same_colour_coins_record = jnp.stack(same_colour_coins_record)
        diff_colour_coins_record = jnp.stack(diff_colour_coins_record)
        coins_collected_info = (same_colour_coins_record, diff_colour_coins_record)

    score_record = jnp.stack(score_record)
    vs_fixed_strats_score_record[0] = jnp.stack(vs_fixed_strats_score_record[0])
    vs_fixed_strats_score_record[1] = jnp.stack(vs_fixed_strats_score_record[1])

    return score_record, vs_fixed_strats_score_record, coins_collected_info


def get_prop_same_coins(ckpts, max_iter_plot=200):
    """
    For each checkpoint, compute the proportion of same-colored coins picked up.
    
    Returns:
        A stacked JAX array of proportions.
    """
    prop_same_coins_record = []
    for ckpt in ckpts:
        score_record, vs_fixed_strats_score_record, coins_collected_info = load_from_checkpoint(
            LOAD_DIR, ckpt, ACTION_SIZE, HIDDEN_SIZE, BATCH_SIZE, INPUT_SIZE, LR_OUT, LR_V, OUTER_OPTIM
        )
        same_colour_coins_record, diff_colour_coins_record = coins_collected_info
        same_colour_coins_record = same_colour_coins_record[:max_iter_plot]
        diff_colour_coins_record = diff_colour_coins_record[:max_iter_plot]
        prop_same_coins = same_colour_coins_record / (same_colour_coins_record + diff_colour_coins_record)
        prop_same_coins_record.append(prop_same_coins)
    return jnp.stack(prop_same_coins_record)


def get_score_individual_ckpt(load_dir, load_prefix, w_coin_record=False):
    """
    Load and compute evaluation scores from a single checkpoint.
    
    Returns:
        Tuple (avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, prop_same_coins)
    """
    score_record, vs_fixed_strats_score_record, coins_collected_info = load_from_checkpoint(
        load_dir, load_prefix, ACTION_SIZE, HIDDEN_SIZE, BATCH_SIZE, INPUT_SIZE, LR_OUT, LR_V, OUTER_OPTIM
    )
    agent1_vs_fixed_strat, agent2_vs_fixed_strat = vs_fixed_strats_score_record
    avg_scores = score_record.mean(axis=1)
    avg_vs_fixed = (agent1_vs_fixed_strat + agent2_vs_fixed_strat) / 2.
    avg_vs_alld = avg_vs_fixed[:, 0]
    avg_vs_allc = avg_vs_fixed[:, 1]
    avg_vs_tft = avg_vs_fixed[:, 2]
    if w_coin_record:
        same_colour_coins, diff_colour_coins = coins_collected_info
        prop_same_coins = same_colour_coins / (same_colour_coins + diff_colour_coins)
        return avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, prop_same_coins

    return avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, None


def get_scores(ckpts, max_iter_plot=200, w_coin_record=False):
    score_record = []
    avg_vs_alld_record = []
    avg_vs_allc_record = []
    avg_vs_tft_record = []
    if w_coin_record:
        coin_record = []
    for i in range(len(ckpts)):
        ckpts_sublist = ckpts[i]
        if isinstance(ckpts_sublist, list):
            score_subrecord = []
            avg_vs_alld_subrecord = []
            avg_vs_allc_subrecord = []
            avg_vs_tft_subrecord = []
            coin_subrecord = []
            for ckpt in ckpts_sublist:
                load_prefix = ckpt
                avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, prop_same_coins = get_score_individual_ckpt(
                    load_dir, load_prefix, w_coin_record=w_coin_record)
                score_subrecord.append(avg_scores)
                avg_vs_alld_subrecord.append(avg_vs_alld)
                avg_vs_allc_subrecord.append(avg_vs_allc)
                avg_vs_tft_subrecord.append(avg_vs_tft)
                coin_subrecord.append(prop_same_coins)
            avg_scores = jnp.concatenate(score_subrecord)[:max_iter_plot]
            avg_vs_alld = jnp.concatenate(avg_vs_alld_subrecord)[:max_iter_plot]
            avg_vs_allc = jnp.concatenate(avg_vs_allc_subrecord)[:max_iter_plot]
            avg_vs_tft = jnp.concatenate(avg_vs_tft_subrecord)[:max_iter_plot]
            if w_coin_record:
                prop_c = jnp.concatenate(coin_subrecord)[:max_iter_plot]

        else:
            load_prefix = ckpts[i]
            avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, prop_c = get_score_individual_ckpt(load_dir, load_prefix, w_coin_record=w_coin_record)

        score_record.append(avg_scores[:max_iter_plot])
        avg_vs_alld_record.append(avg_vs_alld[:max_iter_plot])
        avg_vs_allc_record.append(avg_vs_allc[:max_iter_plot])
        avg_vs_tft_record.append(avg_vs_tft[:max_iter_plot])
        if w_coin_record:
            coin_record.append(prop_c[:max_iter_plot])


    score_record = jnp.stack(score_record)
    avg_vs_alld_record = jnp.stack(avg_vs_alld_record)
    avg_vs_allc_record = jnp.stack(avg_vs_allc_record)
    avg_vs_tft_record = jnp.stack(avg_vs_tft_record)
    if w_coin_record:
        coin_record = jnp.stack(coin_record)
        return score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record, coin_record

    return score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record


def plot_coins_record(ckpts, max_iter_plot, label, z_score=1.96, skip_step=0):

    prop_same_coins_record = get_prop_same_coins(ckpts, max_iter_plot=max_iter_plot)

    plot_with_conf_bounds(prop_same_coins_record, max_iter_plot, len(ckpts), label, skip_step,
                          z_score)

def plot_with_conf_bounds(record, max_iter_plot, num_ckpts, label, skip_step, z_score, use_ax=False, ax=None, linestyle='solid'):
    avg = record.mean(axis=0)

    stdev = jnp.std(record, axis=0)

    upper_conf_bound = avg + z_score * stdev / np.sqrt(
        num_ckpts)
    lower_conf_bound = avg - z_score * stdev / np.sqrt(
        num_ckpts)

    if use_ax:
        assert ax is not None
        ax.plot(np.arange(max_iter_plot) * skip_step, avg,
             label=label, linestyle=linestyle)
        ax.fill_between(np.arange(max_iter_plot) * skip_step, lower_conf_bound,
                     upper_conf_bound, alpha=0.3)

    else:
        plt.plot(np.arange(max_iter_plot) * skip_step, avg,
                 label=label)
        plt.fill_between(np.arange(max_iter_plot) * skip_step, lower_conf_bound,
                         upper_conf_bound, alpha=0.3)


def setup_ipd_plots(titles):
    nfigs = len(titles)
    fig, axs = plt.subplots(1, nfigs, figsize=(5 * (nfigs) + 3, 4))

    for i in range(nfigs):
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Total Number of Outer Steps")
        axs[i].set_ylabel("Score (Average over Agents and Rollout Length)")

    return fig, axs

def setup_coin_plots(titles):
    nfigs = len(titles)
    fig, axs = plt.subplots(1, nfigs, figsize=(4 * (nfigs) + 6, 4))

    axs[0].set_title(titles[0])
    axs[0].set_xlabel("Total Number of Outer Steps")
    axs[0].set_ylabel("Proportion of Same Coins Picked Up")

    for i in range(1, nfigs):
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Total Number of Outer Steps")
        axs[i].set_ylabel("Score (Average over Agents and Rollout Length)")

    return fig, axs

def plot_ipd_results(axs, ckpts, nfigs, max_iter_plot, label, z_score=1.96, skip_step=0, linestyle='solid'):

    score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record = \
        get_scores(ckpts, max_iter_plot=max_iter_plot)

    plot_tup = (score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record)

    for i in range(nfigs):
        plot_with_conf_bounds(plot_tup[i], max_iter_plot, len(ckpts), label,
                              skip_step, z_score, use_ax=True, ax=axs[i], linestyle=linestyle)
        axs[i].legend()

def plot_coin_results(axs, ckpts, nfigs, max_iter_plot, label, z_score=1.96, skip_step=0, linestyle='solid'):

    score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record, prop_same_coins_record = \
        get_scores(ckpts, max_iter_plot=max_iter_plot, w_coin_record=True)

    plot_tup = (prop_same_coins_record, score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record)

    for i in range(nfigs):
        plot_with_conf_bounds(plot_tup[i], max_iter_plot, len(ckpts), label,
                              skip_step, z_score, use_ax=True, ax=axs[i], linestyle=linestyle)
        axs[i].legend()



if plot_coin:
    pola_max_iters = 250 # epochs/n_update
    pola_skip_step = 200 # outer steps
    pola_om_max_iters = pola_max_iters
    pola_om_skip_step = pola_skip_step
    lola_skip_step = 1 # outer steps
    lola_max_iters = pola_max_iters * pola_skip_step // lola_skip_step

    # titles = ("Proportion of Same Coins Picked Up", "Average Score vs Each Other", "Average Score vs Always Defect", "Average Score vs Always Cooperate", "Average Score vs TFT")
    titles = ("Proportion of Same Coins Picked Up", "Average Score vs Each Other", "Average Score vs Always Defect")
    fig, axs = setup_coin_plots(titles)

    # POLA is 200 skip step because 1 inner 1 outer, 100 times = 200 env rollouts per epoch per agent
    plot_coin_results(axs, ckpts_pola, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-DiCE", linestyle='dashed')

    # LOLA is 2 skip step because 1 inner 1 outer, 1 time = 2 env rollouts per epoch per agent
    plot_coin_results(axs, ckpts_lola, nfigs=len(titles), max_iter_plot=lola_max_iters, skip_step=lola_skip_step, label="LOLA-DiCE", linestyle='solid')

    # For OM I'm not counting the env rollouts used for the OM data collection
    plot_coin_results(axs, ckpts_pola_om, nfigs=len(titles), max_iter_plot=pola_om_max_iters, skip_step=pola_om_skip_step, label="POLA-OM", linestyle='dotted')


    # Agents who always cooperate only pick up their own coins, and a coin is picked up on average every 1.5 time steps, so 3 time steps for each agent to pick up a coin, so the maximum expected reward is 1/3 per time step. If you always cooperate against an always defect agent, a coin is picked up on average every 1.5 time steps, but the cooperative agent gets 0 reward half of the time, and for the remaining 25 time steps, the coop agent competes with the always defect agent. If both pick up coin, coop agent gets -1, if coop wins race, coop gets 1, if coop loses coop gets -2. 50% of the time, equal distance, so coop gets -1, then 50% of the time it's a race with expected value -0.5. So average -0.75. So 25 time steps every 1.5 steps you get -0.75. So on 16.666 coins you get -0.75 which is 12.5, then /50 you get -0.25 average. Empirically I find it is -0.26. I think the reasoning is not exactly correct because of the possibility of 2 agents being on the same space.
    x_vals = np.arange(pola_max_iters) * pola_skip_step
    axs[0].plot(x_vals, 1. * np.ones_like(x_vals), label="Always Cooperate",
                linestyle=((0, (3, 1, 1, 1, 1, 1))))
    axs[0].plot(x_vals, 0.5 * np.ones_like(x_vals), label="Always Defect",
                linestyle='dashdot')
    axs[1].plot(x_vals, 1. / 3. * np.ones_like(x_vals), label="Always Cooperate",
                linestyle=((0, (3, 1, 1, 1, 1, 1))))
    axs[1].plot(x_vals, 0 * np.ones_like(x_vals), label="Always Defect",
                linestyle='dashdot')
    axs[2].plot(x_vals, -0.26 * np.ones_like(x_vals), label="Always Cooperate",
                linestyle=((0, (3, 1, 1, 1, 1, 1))))
    axs[2].plot(x_vals, 0 * np.ones_like(x_vals), label="Always Defect",
                linestyle='dashdot')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

else:
    pola_max_iters = 100 # epochs/n_update
    pola_skip_step = 200 # outer steps
    lola_skip_step = 1 # outer steps
    lola_max_iters = pola_max_iters * pola_skip_step // lola_skip_step

    # titles = ("Average Score vs Each Other", "Average Score vs Always Defect", "Average Score vs Always Cooperate", "Average Score vs TFT")
    titles = ("Average Score vs Each Other", "Average Score vs Always Defect")
    fig, axs = setup_ipd_plots(titles)

    plot_ipd_results(axs, ckpts_pola, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-DiCE", linestyle='dashed')
    plot_ipd_results(axs, ckpts_lola, nfigs=len(titles), max_iter_plot=lola_max_iters, skip_step=lola_skip_step, label="LOLA-DiCE", linestyle='solid')
    plot_ipd_results(axs, ckpts_pola_om, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-OM", linestyle='dotted')

    x_vals = np.arange(pola_max_iters) * pola_skip_step
    axs[0].plot(x_vals, 0.33 * np.ones_like(x_vals), label="Always Cooperate", linestyle=((0, (3, 1, 1, 1, 1, 1))))
    axs[0].plot(x_vals, 0 * np.ones_like(x_vals), label="Always Defect", linestyle='dashdot')
    axs[1].plot(x_vals, -0.335 * np.ones_like(x_vals) , label="Always Cooperate", linestyle=((0, (3, 1, 1, 1, 1, 1))))
    axs[1].plot(x_vals, 0 * np.ones_like(x_vals), label="Always Defect", linestyle='dashdot')
    axs[0].legend()
    axs[1].legend()


plt.show()

fig.savefig('fig.png')
