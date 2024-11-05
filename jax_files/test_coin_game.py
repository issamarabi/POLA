"""
test_coin_game.py

Comprehensive tests for the N-agent CoinGame environment defined in coin_game_jax.py.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

# Import the CoinGame and CoinGameState classes from your refactored file
from coin_game_jax import CoinGame, CoinGameState, MOVES


@pytest.mark.parametrize("n_agents, grid_size", [
    (2, 3),
    (3, 5),
    (4, 5),
])
def test_coin_game_init(n_agents, grid_size):
    """
    Test that a CoinGame instance is initialized with the correct attributes.
    """
    env = CoinGame(n_agents=n_agents, grid_size=grid_size)
    assert env.n_agents == n_agents, "Wrong number of agents in environment."
    assert env.grid_size == grid_size, "Wrong grid size in environment."


@pytest.mark.parametrize("n_agents, grid_size", [
    (2, 3),
    (3, 5),
    (4, 5),
])
def test_generate_new_coin_pos(n_agents, grid_size):
    """
    Test the internal _generate_new_coin_pos method:
      - Check that the newly generated coin position is not in any agent's position.
      - Check that the position is within the grid bounds.
    """
    env = CoinGame(n_agents=n_agents, grid_size=grid_size)
    key = random.PRNGKey(0)
    # Sample random agent positions (flattened)
    # We won't require distinct positions for agents here, just test the coin generation.
    agent_positions_flat = random.randint(key, shape=(n_agents,),
                                          minval=0, maxval=grid_size * grid_size)

    # Generate multiple times to ensure coverage
    for i in range(10):
        subkey, key = random.split(key)
        coin_pos = env._generate_new_coin_pos(subkey, agent_positions_flat)
        r, c = coin_pos
        assert 0 <= r < grid_size, "Coin row is out of bounds."
        assert 0 <= c < grid_size, "Coin column is out of bounds."

        # Convert coin_pos to flat
        coin_pos_flat = r * grid_size + c
        # Ensure coin is not in any agent position
        assert not (coin_pos_flat == agent_positions_flat).any(), (
            "Coin position overlaps an agent position!"
        )


@pytest.mark.parametrize("n_agents, grid_size", [
    (2, 3),
    (3, 5),
    (4, 4),
])
def test_reset(n_agents, grid_size):
    """
    Test the reset function:
      - Check shapes of returned state and observation.
      - Ensure no overlap between coin and agent positions.
      - Ensure coin_color is within [0, n_agents).
      - Check that step_count is zero.
    """
    env = CoinGame(n_agents=n_agents, grid_size=grid_size)
    key = random.PRNGKey(42)
    state, obs = env.reset(key)

    # Check shapes of the state
    assert state.agent_positions.shape == (n_agents, 2), "agent_positions shape is incorrect."
    assert state.coin_pos.shape == (2,), "coin_pos shape is incorrect."
    assert isinstance(state.coin_color, jnp.ndarray), "coin_color must be jnp.ndarray."
    assert state.step_count.shape == (), "step_count must be a scalar."

    # Check the obs shape: (2*n_agents * (grid_size^2),)
    expected_obs_size = 2 * n_agents * (grid_size * grid_size)
    assert obs.shape == (expected_obs_size,), "Flattened observation has incorrect shape."

    # No overlap between coin and agent positions
    coin_r, coin_c = state.coin_pos
    coin_flat = coin_r * grid_size + coin_c
    agent_positions_flat = state.agent_positions[:, 0] * grid_size + state.agent_positions[:, 1]
    assert not (coin_flat == agent_positions_flat).any(), "Coin overlaps with an agent position."

    # coin_color in [0, n_agents)
    assert 0 <= state.coin_color < n_agents, "coin_color out of range."

    # step_count is zero
    assert state.step_count == 0, "step_count should be initialized to zero."


def test_state_to_obs():
    """
    Test that state_to_obs produces the correct one-hot layers:
      - n_agents channels for agent positions
      - n_agents channels for coin position color
    """
    n_agents = 4
    grid_size = 5
    env = CoinGame(n_agents=n_agents, grid_size=grid_size)

    # Construct a sample state manually:
    # Let's put agent 0 at (0,0), agent 1 at (1,1), agent 2 at (2,2), agent 3 at (4,4).
    # Put the coin at (3,3), color = 2, step_count=10
    agent_positions = jnp.array([[0,0],[1,1],[2,2],[4,4]], dtype=jnp.int32)
    coin_pos = jnp.array([3,3], dtype=jnp.int32)
    coin_color = jnp.array(2, dtype=jnp.int32)
    step_count = jnp.array(10, dtype=jnp.int32)

    state = CoinGameState(agent_positions, coin_pos, coin_color, step_count)
    obs = env.state_to_obs(state)

    # The obs shape must be (2*n_agents*grid_size*grid_size,)
    expected_size = 2 * n_agents * grid_size * grid_size
    assert obs.shape == (expected_size,)

    # Convert obs back to (2*n_agents, grid_size, grid_size)
    obs_2d = obs.reshape(2 * n_agents, grid_size, grid_size)

    # Check agent channels
    for i in range(n_agents):
        # Exactly one '1' in channel i at agent_positions[i]
        r, c = agent_positions[i]
        channel_vals = obs_2d[i]
        assert channel_vals.sum() == 1.0, f"Agent {i} channel should have exactly one '1'."
        assert channel_vals[r, c] == 1.0, f"Agent {i} position incorrect in channel."

    # Check coin channels
    # Channels [n_agents..2*n_agents-1]
    # coin_color = 2 => that means channel (n_agents + 2) should have 1 at (3,3),
    # others 0 in that same channel
    coin_channel_start = n_agents
    coin_channel_for_color = coin_channel_start + coin_color
    for cc in range(n_agents):
        channel_vals = obs_2d[coin_channel_start + cc]
        if cc == coin_color:
            assert channel_vals.sum() == 1.0, "Coin color channel should have exactly one '1'."
            assert channel_vals[coin_pos[0], coin_pos[1]] == 1.0, "Coin position is incorrect in color channel."
        else:
            assert channel_vals.sum() == 0.0, "Non-coin-color channels should have all zeros."


@pytest.mark.parametrize("n_agents, grid_size", [
    (2, 3),
    (3, 5),
])
def test_step_no_pickup(n_agents, grid_size):
    """
    Test the step function in a situation where no agent picks up the coin:
      - Ensure positions update correctly.
      - Ensure coin remains the same.
      - Ensure reward is all zeros.
      - Ensure step_count increments by 1.
    """
    env = CoinGame(n_agents=n_agents, grid_size=grid_size)
    key = random.PRNGKey(0)
    state, _ = env.reset(key)

    # Guarantee no pickup: move all agents away from the coin deliberately
    # We'll get the moves towards coin and flip them to be away
    moves_towards = env.get_moves_towards_coin(state)
    actions = env.get_moves_away_from_coin(moves_towards)

    key_step, _ = random.split(key)
    new_state, new_obs, rewards, _ = env.step(state, actions, key_step)

    # Check positions vs. naive shift
    expected_positions = (state.agent_positions + MOVES[actions]) % grid_size
    assert jnp.all(new_state.agent_positions == expected_positions), "Agent positions did not update correctly."

    # No coin pickup => coin stays same
    assert jnp.all(new_state.coin_pos == state.coin_pos), "Coin position should remain the same if not picked up."
    assert new_state.coin_color == state.coin_color, "Coin color should remain the same if not picked up."

    # All rewards should be zero
    assert jnp.all(rewards == 0), "Rewards should be all zeros if no coin is picked up."

    # step_count should increment
    assert new_state.step_count == state.step_count + 1, "Step count did not increment by 1."

    # Check new_obs shape
    expected_obs_size = 2 * n_agents * (grid_size * grid_size)
    assert new_obs.shape == (expected_obs_size,), "Flattened observation has incorrect shape after step."


def test_step_pickup_single_agent():
    """
    Test the step function in a scenario where exactly one agent picks up the coin:
      - That agent gets +1 reward.
      - The coin_color agent (if it is not the same that picked up) gets -2.
      - A new coin is generated.
      - coin_color is re-sampled in [0..n_agents).
    """
    n_agents = 3
    grid_size = 4
    env = CoinGame(n_agents=n_agents, grid_size=grid_size)

    # Construct a state such that:
    #   agent 0 is at (0,0)
    #   agent 1 is at (1,1)
    #   agent 2 is at (2,2)
    #   coin is at (0,1)
    #   coin_color = 1
    # We want agent 0 to pick up the coin in one move.
    agent_positions = jnp.array([[0,0],[1,1],[2,2]], dtype=jnp.int32)
    coin_pos = jnp.array([0,1], dtype=jnp.int32)
    coin_color = jnp.array(1, dtype=jnp.int32)
    step_count = jnp.array(5, dtype=jnp.int32)
    state = CoinGameState(agent_positions, coin_pos, coin_color, step_count)

    # Action for agent 0: move right (0), this should pick up coin at (0,1).
    # Action for agent 1: move left  (1) => doesn't matter
    # Action for agent 2: move left  (1) => doesn't matter
    actions = jnp.array([0, 1, 1], dtype=jnp.int32)

    key = random.PRNGKey(123)
    new_state, new_obs, rewards, _ = env.step(state, actions, key)

    # Check new agent positions
    expected_positions = jnp.array([
        [0,1],  # agent 0 => picked up coin
        [1,0],
        [2,1],
    ], dtype=jnp.int32)
    assert jnp.all(new_state.agent_positions == expected_positions)

    # Agent 0 picks up the coin, coin_color = 1 => that means:
    #   agent 0 => +1
    #   agent 1 => -2 (since agent 1 is the coin_color but did not pick it up)
    #   agent 2 => 0
    expected_rewards = jnp.array([1.0, -2.0, 0.0], dtype=jnp.float32)
    assert jnp.allclose(rewards, expected_rewards), f"Rewards mismatch: got {rewards}, expected {expected_rewards}."

    # The coin must have been re-generated => new_state.coin_pos != old coin_pos
    assert not jnp.all(new_state.coin_pos == coin_pos), "Coin should be respawned at a new position."

    # The coin_color is re-sampled in [0, n_agents)
    assert 0 <= new_state.coin_color < n_agents, "New coin_color is out of range!"

    # step_count increments
    assert new_state.step_count == state.step_count + 1

    # Check the new_obs shape
    expected_obs_size = 2 * n_agents * (grid_size * grid_size)
    assert new_obs.shape == (expected_obs_size,)


def test_step_pickup_multiple_agents():
    """
    Test the step function where multiple agents simultaneously pick up the coin:
      - Each agent that picks up gets +1.
      - If coin_color != any of those agents, that coin_color agent gets -2 for *each* agent that picks up.
        (Implementation sums the -2 across all agents who pick up if coin_color is different.)
      - A new coin is generated exactly once.
    """
    n_agents = 3
    grid_size = 3
    env = CoinGame(n_agents=n_agents, grid_size=grid_size)

    # Place two agents at positions that can both move onto coin in a single step.
    agent_positions = jnp.array([[0, 0], [0, 2], [2, 2]], dtype=jnp.int32)
    coin_pos = jnp.array([0, 1], dtype=jnp.int32)
    coin_color = jnp.array(2, dtype=jnp.int32)  # coin belongs to agent 2
    step_count = jnp.array(0, dtype=jnp.int32)
    state = CoinGameState(agent_positions, coin_pos, coin_color, step_count)

    # If agent 0 moves right (0), it lands on (0,1).
    # If agent 1 moves left (1), it lands on (0,1).
    # Agent 2 does something else, e.g. up (3), doesn't matter.
    actions = jnp.array([0, 1, 3], dtype=jnp.int32)

    key = random.PRNGKey(999)
    new_state, new_obs, rewards, _ = env.step(state, actions, key)

    # Agents 0 and 1 pick up => each gets +1
    # coin_color = 2 => that agent (2) gets -2 for *each* agent that picks up.
    # Because two agents pick up, agent 2 => -2 for each, which sums to -4 in total.
    expected_rewards = jnp.array([1.0, 1.0, -4.0], dtype=jnp.float32)
    assert jnp.allclose(rewards, expected_rewards), f"Got {rewards}, expected {expected_rewards}."

    # coin is re-generated
    assert not jnp.all(new_state.coin_pos == coin_pos), "Coin should be respawned at a new position."

    # coin_color re-sampled
    assert 0 <= new_state.coin_color < n_agents, "New coin_color out of range."

    # step_count increments
    assert new_state.step_count == 1

    # Check shape of new_obs
    expected_obs_size = 2 * n_agents * (grid_size * grid_size)
    assert new_obs.shape == (expected_obs_size,)


def test_get_moves_towards_coin():
    """
    Test get_moves_towards_coin:
      - Construct a scenario with known distances.
      - Check that the returned moves indeed reduce distance to the coin.
    """
    n_agents = 3
    grid_size = 5
    env = CoinGame(n_agents=n_agents, grid_size=grid_size)

    # Let's place a coin at (2,2).
    # Agents:
    #   0 at (0,0),
    #   1 at (4,4),
    #   2 at (2,0).
    agent_positions = jnp.array([[0,0],[4,4],[2,0]], dtype=jnp.int32)
    coin_pos = jnp.array([2,2], dtype=jnp.int32)
    coin_color = jnp.array(1, dtype=jnp.int32)
    step_count = jnp.array(10, dtype=jnp.int32)
    state = CoinGameState(agent_positions, coin_pos, coin_color, step_count)

    moves = env.get_moves_towards_coin(state)
    # moves has shape [n_agents], each in {0,1,2,3} => right, left, down, up

    # Let's see if these moves are reasonable:
    #   agent 0: (0,0) => coin (2,2).  The minimal vertical distance is 2, horizontal is 2.
    #              The tie means "prefer vertical" in the docstring? Or you might have "prefer horizontal".
    #              The method says: "Compare minimal distances; if min_horiz > min_vert => move horizontally,
    #              else move vertically. If equal, prefer vertical."
    #              So for agent 0: min_vert_dist = 2, min_horiz_dist = 2 => they are equal => prefer vertical => move 2=down
    #   agent 1: (4,4) => coin (2,2).  horizontal dist = 3 or 1 (since (2 - 4) % 5 = 3, (4 - 2) % 5=2?), let's be precise:
    #       horiz_dist_right = (2-4)%5=3, horiz_dist_left = (4-2)%5=2 => min_horiz=2 => direction=1=left
    #       vert_dist_down = (2-4)%5=3, vert_dist_up=(4-2)%5=2 => min_vert=2 => direction=3=up
    #       So min_vert=2, min_horiz=2 => they are equal => prefer vertical => up=3
    #   agent 2: (2,0) => coin(2,2).  vertical dist=0, horizontal dist=2 => min_vert=0 < min_horiz=2 => move horizontally?
    #              The rule says: "move_horiz = (min_horiz_dist>min_vert_dist)" => if 2>0 => True => move horizontally.
    #              Among horizontally, (coin_c - ac)%5=2 => right=0 is smaller than left=3 => so 0=right
    # So we expect moves = [2 (down), 3 (up), 0 (right)]
    expected_moves = jnp.array([2, 3, 0], dtype=jnp.int32)

    assert jnp.all(moves == expected_moves), f"Moves mismatch. Got {moves}, expected {expected_moves}"


def test_get_moves_away_from_coin():
    """
    Test get_moves_away_from_coin:
      - Given an array of moves_towards_coin, check the output is the opposite.
    """
    env = CoinGame(n_agents=4, grid_size=5)
    # moves_toward in {0->1, 1->0, 2->3, 3->2}
    moves_toward = jnp.array([0, 1, 2, 3])
    moves_away = env.get_moves_away_from_coin(moves_toward)

    # We expect [1, 0, 3, 2]
    expected = jnp.array([1, 0, 3, 2])
    assert jnp.all(moves_away == expected), f"Got {moves_away}, expected {expected}"

    # Try random arrays
    key = random.PRNGKey(0)
    for _ in range(5):
        moves_rand = random.randint(key, shape=(4,), minval=0, maxval=4)
        key, _ = random.split(key)
        moves_opp = env.get_moves_away_from_coin(moves_rand)
        # Check that each entry is indeed the opposite
        for i in range(4):
            if moves_rand[i] == 0:
                assert moves_opp[i] == 1
            elif moves_rand[i] == 1:
                assert moves_opp[i] == 0
            elif moves_rand[i] == 2:
                assert moves_opp[i] == 3
            elif moves_rand[i] == 3:
                assert moves_opp[i] == 2


@pytest.mark.parametrize("n_agents, grid_size", [
    (2, 3),
    (4, 5),
])
def test_get_coop_actions(n_agents, grid_size):
    """
    Test get_coop_actions:
      - The coin_color agent moves TOWARD the coin
      - Every other agent moves AWAY from the coin
    """
    env = CoinGame(n_agents=n_agents, grid_size=grid_size)
    key = random.PRNGKey(123)
    state, _ = env.reset(key)

    # The coin_color is in state.coin_color
    # We'll compute the standard "toward" moves for everyone, then "away" moves
    # and compare to see if get_coop_actions is picking them appropriately.
    moves_toward = env.get_moves_towards_coin(state)
    moves_away = env.get_moves_away_from_coin(moves_toward)
    coop_moves = env.get_coop_actions(state)

    for i in range(n_agents):
        if i == state.coin_color:
            assert coop_moves[i] == moves_toward[i], f"Agent {i} is coin_color but not moving toward coin!"
        else:
            assert coop_moves[i] == moves_away[i], f"Agent {i} is not coin_color but not moving away from coin!"

    # Edge: If the coin_color agent is already on the coin, the "toward" move might be arbitrary,
    # but the function still systematically picks it for that agent. We just ensure no crash.


if __name__ == "__main__":
    pytest.main([__file__])
