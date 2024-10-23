import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

# Define constants for possible moves.
MOVES = jax.device_put(
    jnp.array(
        [
            [0, 1],   # right
            [0, -1],  # left
            [1, 0],   # down
            [-1, 0],  # up
        ]
    )
)

class CoinGameState(NamedTuple):
    """
    Represents the state of the N-agent CoinGame.

    Attributes:
    - agent_positions: jnp.ndarray of shape [n_agents, 2]
    - coin_pos: 2D position of the coin, shape [2]
    - coin_color: scalar integer in [0, n_agents), meaning the coin "belongs" to that agent
    - step_count: Number of steps taken in the current episode.
    """
    agent_positions: jnp.ndarray
    coin_pos: jnp.ndarray
    coin_color: jnp.ndarray
    step_count: jnp.ndarray


class CoinGame:
    """
    Defines an N-agent CoinGame environment on a grid of size (grid_size x grid_size).

    """

    def __init__(self, n_agents=2, grid_size=3):
        """
        Args:
        - n_agents: number of agents (>= 2)
        - grid_size: dimension of the square grid
        """
        self.n_agents = n_agents
        self.grid_size = grid_size

    def _generate_new_coin_pos(self, key: jnp.ndarray, agent_positions_flat: jnp.ndarray) -> jnp.ndarray:
        """
        Samples a random position for the coin that does not overlap with any agent's position.
        Args:
        - key: PRNGKey
        - agent_positions_flat: shape [n_agents], each is the flattened index row*grid_size + col

        Returns: shape [2], row and col
        """
        # We exclude the agent positions from the possible coin positions
        # Maximum of (grid_size^2 - n_agents) possible positions.
        # We'll do a simple approach: sample among all positions, then re-sample if it’s an agent’s position

        # Sample an integer in [0, grid_area - n_agents] then shift it up by #excluded slots
        coin_key, key = jax.random.split(key, 2)
        max_val_for_coin = (self.grid_size ** 2) - jnp.unique(agent_positions_flat).size

        coin_pos_flat = jax.random.randint(coin_key, shape=(1,), minval=0, maxval=max_val_for_coin)
        # Now offset if it is in an excluded region
        def shift_if_needed_scan(pos_flat, agent_pos):
            pos_flat = pos_flat + (pos_flat >= agent_pos)
            return pos_flat, None

        coin_pos_flat, _ = jax.lax.scan(shift_if_needed_scan, coin_pos_flat, jnp.sort(jnp.unique(agent_positions_flat)))
        coin_pos = jnp.stack(
            (coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size)
        ).squeeze(-1)
        return coin_pos

    def reset(self, key: jnp.ndarray) -> Tuple[CoinGameState, jnp.ndarray]:
        """
        Resets the environment:
          - Places each of the n_agents in a random distinct cell
          - Places the coin in a random cell distinct from agent positions
          - Randomly picks a coin_color in [0, n_agents)

        Returns:
          - state: CoinGameState
          - obs: jnp.ndarray  Flattened observation
        """
        # Randomly place each agent
        # For simplicity, let each agent's position be drawn i.i.d. from [0, grid_size^2)
        # If we want to require strictly distinct positions, we can add logic similar to the coin logic.
        subkeys = jax.random.split(key, 1 + self.n_agents)
        key_agent_positions = subkeys[0]
        agent_pos_flat = []
        for i in range(self.n_agents):
            pos_flat_i = jax.random.randint(
                subkeys[i], shape=(1,), minval=0, maxval=self.grid_size**2
            )
            agent_pos_flat.append(pos_flat_i[0])
        agent_pos_flat = jnp.array(agent_pos_flat)  # shape [n_agents]
        agent_positions = jnp.stack(
            (agent_pos_flat // self.grid_size, agent_pos_flat % self.grid_size), axis=-1
        )  # shape [n_agents, 2]

        # Place the coin
        key_coin, key_color = jax.random.split(key_agent_positions, 2)
        coin_pos = self._generate_new_coin_pos(key_coin, agent_pos_flat)

        # coin_color in [0..n_agents)
        coin_color = jax.random.randint(key_color, shape=(), minval=0, maxval=self.n_agents)

        step_count = jnp.zeros((), dtype=jnp.int32)

        state = CoinGameState(agent_positions, coin_pos, coin_color, step_count)
        obs = self.state_to_obs(state)
        return state, obs

    def state_to_obs(self, state: CoinGameState) -> jnp.ndarray:
        """
        Builds a channels x grid_size x grid_size representation, then flattens it.
        We use the first n_agents channels to mark agent positions, then n_agents channels
        to mark the coin color at the coin position.

        => final shape = (2*n_agents, grid_size, grid_size), flattened to size 2*n_agents*(grid_size^2).
        """
        n_agents = self.n_agents
        grid_size = self.grid_size

        # Initialize channels
        #   channel i   => agent i positions
        #   channel n_agents + j => j==coin_color means that channel is 1 at coin pos
        obs = jnp.zeros((2 * n_agents, grid_size, grid_size))

        # Fill in agent positions
        def mark_agent(obs_acc, agent_idx):
            row, col = state.agent_positions[agent_idx]
            return obs_acc.at[agent_idx, row, col].set(1.0)

        obs = jax.lax.fori_loop(0, n_agents, lambda i, o: mark_agent(o, i), obs)

        # Fill in coin position
        coin_r, coin_c = state.coin_pos
        coin_col = state.coin_color  # integer in [0..n_agents)
        obs = obs.at[n_agents + coin_col, coin_r, coin_c].set(1.0)

        # Flatten
        obs = obs.reshape(-1)
        return obs

    def step(self, state: CoinGameState, action_0: int, action_1: int, subkey: jnp.ndarray) -> Tuple[jnp.ndarray, list]:
        """
        Execute a step in the game based on the actions of the two players.

        Args:
        - state: Current state of the game.
        - action_0: Action taken by the red player.
        - action_1: Action taken by the blue player.
        - subkey: Random seed.

        Returns:
        - new_state: New state of the game after the step.
        - obs: Observation representation of the new state.
        - rewards: Tuple containing rewards for red and blue players.
        - matches: Tuple indicating matches between players and coins.
        """
        new_red_pos = (state.red_pos + MOVES[action_0]) % 3
        new_blue_pos = (state.blue_pos + MOVES[action_1]) % 3

        is_red_coin = state.is_red_coin[0]
        zero_rew = jnp.zeros(1)

        # Calculate rewards based on player and coin positions
        red_red_matches = jnp.all(new_red_pos == state.coin_pos, axis=-1) * is_red_coin
        red_blue_matches = jnp.all(new_red_pos == state.coin_pos, axis=-1) * (1 - is_red_coin)
        blue_red_matches = jnp.all(new_blue_pos == state.coin_pos, axis=-1) * is_red_coin
        blue_blue_matches = jnp.all(new_blue_pos == state.coin_pos, axis=-1) * (1 - is_red_coin)

        red_reward = jnp.where(red_red_matches | red_blue_matches, zero_rew + 1, zero_rew)
        blue_reward = jnp.where(blue_red_matches | blue_blue_matches, zero_rew + 1, zero_rew)
        red_reward += jnp.where(blue_red_matches, zero_rew - 2, zero_rew)
        blue_reward += jnp.where(red_blue_matches, zero_rew - 2, zero_rew)

        # Check if a new coin needs to be generated
        need_new_coins = ((red_red_matches + red_blue_matches + blue_red_matches + blue_blue_matches) > 0)
        flipped_is_red_coin = 1 - state.is_red_coin
        new_is_red_coin = need_new_coins * flipped_is_red_coin + (1 - need_new_coins) * state.is_red_coin

        new_red_pos_flat = new_red_pos[0] * self.grid_size + new_red_pos[1]
        new_blue_pos_flat = new_blue_pos[0] * self.grid_size + new_blue_pos[1]

        generated_coins = self.generate_coins(subkey, new_red_pos_flat, new_blue_pos_flat)
        new_coin_pos = need_new_coins * generated_coins + (1-need_new_coins) * state.coin_pos

        step_count = state.step_count + 1

        new_state = CoinGameState(new_red_pos, new_blue_pos, new_coin_pos, new_is_red_coin, step_count)
        obs = self.state_to_obs(new_state)

        red_reward = red_reward.squeeze(-1)
        blue_reward = blue_reward.squeeze(-1)

        return new_state, obs, (red_reward, blue_reward), (red_red_matches, red_blue_matches, blue_red_matches, blue_blue_matches)


    def get_moves_shortest_path_to_coin(self, state, red_agent_perspective=True):
        """
        Calculate the move towards the shortest path to the coin in a grid environment.

        This function computes the shortest path for an agent (either red or blue) to a coin
        located on a grid. It evaluates the horizontal and vertical distances from the agent to the coin
        and selects the move (action) that minimally reduces this distance.

        Parameters:
        - state (object): The current state of the environment, which includes the positions
                          of the agent and the coin.
        - red_agent_perspective (bool): A flag indicating whether the calculation is from the
                                        perspective of the red agent. If False, the calculation
                                        is done for the blue agent.

        Returns:
        - actions (jax.numpy.ndarray): An array of integers representing the action(s) leading
                                       towards the shortest path to the coin. The actions are encoded as:
                                       0 - Move right, 1 - Move left, 2 - Move down, 3 - Move up.

        Note:
        - The grid size of the environment is taken into account, and the function assumes a toroidal
          (wrap-around) topology.
        - The calculation prioritizes horizontal movement (left/right) over vertical movement (up/down).
        """
        # Choose the agent's position based on perspective
        agent_pos = state.red_pos if red_agent_perspective else state.blue_pos

        # Calculate horizontal and vertical distances to the coin, modular with the grid size
        horiz_dist_right = (state.coin_pos[:, 1] - agent_pos[:, 1]) % self.grid_size
        horiz_dist_left = (agent_pos[:, 1] - state.coin_pos[:, 1]) % self.grid_size
        vert_dist_down = (state.coin_pos[:, 0] - agent_pos[:, 0]) % self.grid_size
        vert_dist_up = (agent_pos[:, 0] - state.coin_pos[:, 0]) % self.grid_size

        # Initialize actions with a default value (e.g., 0)
        actions = jnp.zeros_like(agent_pos[:, 0])

        # Determine the action (move) based on the shortest path to the coin
        actions = jnp.where(horiz_dist_right < horiz_dist_left, 0, actions)  # Move right
        actions = jnp.where(horiz_dist_left < horiz_dist_right, 1, actions)  # Move left
        actions = jnp.where(vert_dist_down < vert_dist_up, 2, actions)      # Move down
        actions = jnp.where(vert_dist_up < vert_dist_down, 3, actions)      # Move up

        return actions


    def get_moves_away_from_coin(self, moves_towards_coin: jnp.ndarray) -> jnp.ndarray:
         """
         Get the move that takes the agent away from the coin.
         Args:
         - moves_towards_coin: The move that brings the agent closer to the coin.
         Returns:
         - opposite_moves: The move that takes the agent away from the coin.
         """
         opposite_moves = jnp.zeros_like(moves_towards_coin)
         opposite_moves = jnp.where(moves_towards_coin == 0, 1, opposite_moves)
         opposite_moves = jnp.where(moves_towards_coin == 1, 0, opposite_moves)
         opposite_moves = jnp.where(moves_towards_coin == 2, 3, opposite_moves)
         opposite_moves = jnp.where(moves_towards_coin == 3, 2, opposite_moves)

         return opposite_moves


    def get_coop_action(self, state, red_agent_perspective=True) -> jnp.ndarray:
        """
        Get the cooperative action for the agent.

        Args:
        - state: Current state of the game.
        - red_agent_perspective: If True, consider the red agent's perspective, otherwise consider the blue agent's.

        Returns:
        - coop_moves: The cooperative move for the agent.
        """
        moves_towards_coin = self.get_moves_shortest_path_to_coin(state, red_agent_perspective=red_agent_perspective)
        moves_away_from_coin = self.get_moves_away_from_coin(moves_towards_coin)

        is_my_coin = state.is_red_coin if red_agent_perspective else 1 - state.is_red_coin
        is_my_coin = is_my_coin.squeeze(-1)

        # Determine the cooperative move based on the coin's color.
        coop_moves = jnp.where(is_my_coin == 1, moves_towards_coin, -1)
        coop_moves = jnp.where(is_my_coin == 0, moves_away_from_coin, coop_moves)

        return coop_moves

