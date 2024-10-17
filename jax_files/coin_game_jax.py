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
    Defines the CoinGame environment.

    Attributes:
    - grid_size: Size of the grid on which the game is played.
    """
    def __init__(self, grid_size=DEFAULT_GRID_SIZE):
        self.grid_size = grid_size

    def generate_coins(self, random_key: jnp.ndarray, red_pos_flat: int, blue_pos_flat: int) -> jnp.ndarray:
        """
        Generate a random position for the coin such that it doesn't overlap with the players' positions.

        Args:
        - random_key: Random seed.
        - red_pos_flat: Flattened position of the red player.
        - blue_pos_flat: Flattened position of the blue player.

        Returns:
        - coin_pos: 2D position of the coin.
        """
        random_key, key_for_max_val, key_for_coin_pos = jax.random.split(random_key, 3)
        max_val_for_coin = (self.grid_size ** 2) - 2 + (red_pos_flat == blue_pos_flat)
        coin_pos_flat = jax.random.randint(key_for_coin_pos, shape=(1,), minval=0, maxval=max_val_for_coin)
        coin_pos_flat += (coin_pos_flat >= jnp.min(jnp.array([red_pos_flat, blue_pos_flat])))
        coin_pos_flat += jnp.logical_and(coin_pos_flat >= jnp.max(jnp.array([red_pos_flat, blue_pos_flat])), 
                                         red_pos_flat != blue_pos_flat)
        coin_pos = jnp.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size)).squeeze(-1)
        return coin_pos

    def reset(self, random_key: jnp.ndarray) -> Tuple[CoinGameState, jnp.ndarray]:
        """
        Reset the game to its initial state.

        Args:
        - random_key: Random seed.

        Returns:
        - state: Initial state of the game.
        - obs: Initial observation.
        """
        random_key, key_for_red, key_for_blue, key_for_coin = jax.random.split(random_key, 4)
        red_pos_flat = jax.random.randint(key_for_red, shape=(1,), minval=0, maxval=self.grid_size ** 2)
        red_pos = jnp.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size)).squeeze(-1)
        blue_pos_flat = jax.random.randint(key_for_blue, shape=(1,), minval=0, maxval=self.grid_size ** 2)
        blue_pos = jnp.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size)).squeeze(-1)
        coin_pos = self.generate_coins(key_for_coin, red_pos_flat[0], blue_pos_flat[0])
        step_count = jnp.zeros(1)
        is_red_coin = jax.random.randint(key_for_coin, shape=(1,), minval=COIN_RED, maxval=COIN_BLUE+1)
        state = CoinGameState(red_pos, blue_pos, coin_pos, is_red_coin, step_count)
        obs = self.state_to_obs(state)
        return state, obs



    def state_to_obs(self, state: CoinGameState) -> jnp.ndarray:
        """
        Convert the game state to an observation.

        Args:
        - state: Current state of the game.

        Returns:
        - obs: Observation representation of the state.
        """
        is_red_coin = state.is_red_coin[0]
        obs = jnp.zeros((4, 3, 3))
        obs = obs.at[0, state.red_pos[0], state.red_pos[1]].set(1.0)  # Mark red player's position
        obs = obs.at[1, state.blue_pos[0], state.blue_pos[1]].set(1.0)  # Mark blue player's position
        obs = obs.at[2, state.coin_pos[0], state.coin_pos[1]].set(is_red_coin)  # Mark red coin's position
        obs = obs.at[3, state.coin_pos[0], state.coin_pos[1]].set(1.0 - is_red_coin)  # Mark blue coin's position
        obs = obs.reshape(36)
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

