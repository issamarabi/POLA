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

    def step(
        self,
        state: CoinGameState,
        actions: jnp.ndarray,  # shape [n_agents], each in {0,1,2,3} = {right,left,down,up}
        key: jnp.ndarray
    ) -> Tuple[CoinGameState, jnp.ndarray, jnp.ndarray]:
        """
        Moves each agent according to actions, checks if anyone picks up coin, updates state.

        Returns:
        - new_state: CoinGameState
        - obs: jnp.ndarray (flattened)
        - rewards: shape [n_agents]
        """
        n_agents = self.n_agents
        grid_size = self.grid_size

        # 1) Move the agents
        # actions is shape [n_agents], each in [0..3]
        # new_positions[i] = old_positions[i] + MOVES[actions[i]] (mod grid_size)
        new_positions = (state.agent_positions + MOVES[actions]) % grid_size

        # 2) Check which agents pick up the coin
        coin_r, coin_c = state.coin_pos
        picks_up = jnp.all(new_positions == jnp.array([coin_r, coin_c]), axis=-1)  # shape [n_agents], True/False

        # If picks_up[i] is True, agent i picks up the coin.
        # Original 2-agent code gave agent i: +1,
        #   if coin_color != i => coin_color's reward -2
        #   then a new coin is generated with color flipped.
        # For N-agent, we do:
        #   agent i gets +1
        #   if i != coin_color => coin_color gets -2
        #   if ANY agent picks up => we spawn a new coin in random location
        #   coin_color is re-sampled randomly in [0..n_agents)

        zero_rew = jnp.zeros((n_agents,), dtype=jnp.float32)

        # 2a) base reward is zero
        rewards = zero_rew

        # 2b) for each i that picks up:
        #     i => +1
        #     coin_color => -2 (if different from i)
        # Because multiple agents can pick up simultaneously,
        # we sum up the +1 for each agent that picks up, and sum up the -2 once per agent picking up
        # if that agent is not the coin_color.
        # picks_up is [n_agents], so we can do:
        picked_up_indices = jnp.nonzero(picks_up, size=n_agents, fill_value=-1)[0]  # array of shape [n_agents], with -1 for fillers
        # we will loop over those indices (except the -1s) to add the rewards
        def pick_body(carry, idx):
            rewards_acc = carry
            agent_idx = idx
            # agent_idx might be -1 if unused, so we can do a check:
            cond = (agent_idx >= 0)
            # +1 for that agent
            rewards_acc = jnp.where(cond, rewards_acc.at[agent_idx].add(1.0), rewards_acc)
            # if coin_color != agent_idx => coin_color gets -2
            different = (agent_idx != state.coin_color)
            rewards_acc = jnp.where(
                cond & different,
                rewards_acc.at[state.coin_color].add(-2.0),
                rewards_acc
            )
            return rewards_acc, None

        rewards, _ = jax.lax.scan(pick_body, rewards, picked_up_indices)

        # 3) If ANY agent picks up => we spawn a new coin
        any_pickup = picks_up.any()
        def new_coin_f():
            # re-sample coin pos
            new_positions_flat = new_positions[:, 0] * grid_size + new_positions[:, 1]
            new_coin_pos = self._generate_new_coin_pos(key, new_positions_flat)
            # re-sample coin color
            coin_color_key = jax.random.fold_in(key, jnp.sum(state.step_count))
            new_coin_color = jax.random.randint(coin_color_key, shape=(), minval=0, maxval=n_agents)
            return (new_coin_pos, new_coin_color)

        def old_coin_f():
            return (state.coin_pos, state.coin_color)

        new_coin_pos, new_coin_color = jax.lax.cond(
            any_pickup,
            new_coin_f,
            old_coin_f
        )

        # 4) Build the new state
        new_step_count = state.step_count + 1
        new_state = CoinGameState(
            agent_positions=new_positions,
            coin_pos=new_coin_pos,
            coin_color=new_coin_color,
            step_count=new_step_count
        )
        # 5) Build obs
        obs = self.state_to_obs(new_state)
        return new_state, obs, rewards


    def get_moves_towards_coin(self, state: CoinGameState) -> jnp.ndarray:
        """
        Returns a [n_agents] array of actions in {0,1,2,3}, each action
        makes the corresponding agent move closer (mod wrap-around) to the coin.

        This generalizes the old 'get_moves_shortest_path_to_coin(...)'
        but returns one move for *every* agent i in [0..n_agents-1].
        """
        grid_size = self.grid_size
        # Unpack
        agent_positions = state.agent_positions  # shape [n_agents, 2]
        coin_r, coin_c = state.coin_pos          # shape []

        # Each agent i: position (ar_i, ac_i)
        ar = agent_positions[:, 0]
        ac = agent_positions[:, 1]

        # Mod distances for horizontal axis (c)
        # "horiz_dist_right" = how far agent i must go right to reach coin_c
        horiz_dist_right = (coin_c - ac) % grid_size
        horiz_dist_left  = (ac - coin_c) % grid_size

        # Mod distances for vertical axis (r)
        vert_dist_down = (coin_r - ar) % grid_size
        vert_dist_up   = (ar - coin_r) % grid_size

        # We pick the smallest direction among {left, right, up, down}
        # Priority order could be changed if you want tie-break differently.
        # For each agent i, we will produce an action in {0,1,2,3}
        # We'll do it with a direct jnp.where logic or by comparing distances.

        # Start by defaulting to 0=right
        actions = jnp.zeros((self.n_agents,), dtype=jnp.int32)

        # If left is strictly smaller than right/down/up => set action=1
        actions = jnp.where(
            (horiz_dist_left < horiz_dist_right) & 
            (horiz_dist_left < vert_dist_down)  &
            (horiz_dist_left < vert_dist_up),
            1,  # left
            actions
        )
        # If down is strictly smaller than right/up => set action=2
        # but note we must also compare to left in the correct order if you want a single winner
        actions = jnp.where(
            (vert_dist_down < horiz_dist_right) &
            (vert_dist_down < vert_dist_up)     &
            (vert_dist_down < horiz_dist_left),
            2,  # down
            actions
        )
        # If up is strictly smaller than right => set action=3
        # likewise compare to left/down
        actions = jnp.where(
            (vert_dist_up < horiz_dist_right) &
            (vert_dist_up < horiz_dist_down)  &
            (vert_dist_up < horiz_dist_left),
            3,  # up
            actions
        )

        # If none of the above conditions are true, it remains 0 (right).
        # This is just one tie-break pattern.

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

