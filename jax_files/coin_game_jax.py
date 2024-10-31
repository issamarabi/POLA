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
        - key:       PRNGKey (shape [2], uint32)
        - agent_positions_flat: shape [n_agents], each is flattened index row*grid_size + col in [0, grid_size^2)

        Returns: shape [2], row and col
        """
        grid_area = self.grid_size * self.grid_size

        # A small function to check if the chosen position overlaps any agent.
        def is_occupied(pos_flat: int) -> bool:
            return (pos_flat == agent_positions_flat).any()

        # The loop's condition: keep sampling if the coin position is occupied.
        def cond_fun(carry):
            pos, _, _ = carry
            return is_occupied(pos)

        # The loop body: re-sample a new position.
        def body_fun(carry):
            _, tries, loop_key = carry
            loop_key, subkey = jax.random.split(loop_key)
            new_pos = jax.random.randint(subkey, shape=(), minval=0, maxval=grid_area)
            return (new_pos, tries + 1, loop_key)

        # Initial state: sample once and check.
        key, subkey = jax.random.split(key)
        init_pos = jax.random.randint(subkey, shape=(), minval=0, maxval=grid_area)
        init_state = (init_pos, jnp.array(0, dtype=jnp.int32), key)

        # Run the while_loop until we find a free cell.
        final_pos, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)

        # Convert flat index to (row, col).
        coin_pos = jnp.array([final_pos // self.grid_size, final_pos % self.grid_size], dtype=jnp.int32)
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
        - info: None
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
        return new_state, obs, rewards, None


    def get_moves_towards_coin(self, state: CoinGameState) -> jnp.ndarray:
        """
        Returns a [n_agents] array of actions in {0,1,2,3}, each action
        makes the corresponding agent move closer (mod wrap-around) to the coin.

        The move selection logic for each agent is as follows:
        1. Compute the minimal horizontal distance and determine the direction (left/right).
        2. Compute the minimal vertical distance and determine the direction (up/down).
        3. Compare the minimal distances:
        - If min_horiz_dist > min_vert_dist, move horizontally.
        - Else, move vertically.
        - If distances are equal, prefer vertical movement.
        """
        grid_size = self.grid_size

        # Unpack agent positions and coin position
        agent_positions = state.agent_positions  # shape [n_agents, 2]
        coin_r, coin_c = state.coin_pos          # shape [2,]

        # Split agent positions into rows and columns
        ar = agent_positions[:, 0]  # [n_agents]
        ac = agent_positions[:, 1]  # [n_agents]

        # Compute horizontal distances
        horiz_dist_right = (coin_c - ac) % grid_size
        horiz_dist_left  = (ac - coin_c) % grid_size
        min_horiz_dist = jnp.minimum(horiz_dist_right, horiz_dist_left)

        # Determine horizontal direction: 0=right, 1=left
        horiz_dir = jnp.where(horiz_dist_right < horiz_dist_left, 0, 1)

        # Compute vertical distances
        vert_dist_down = (coin_r - ar) % grid_size
        vert_dist_up   = (ar - coin_r) % grid_size
        min_vert_dist = jnp.minimum(vert_dist_down, vert_dist_up)

        # Determine vertical direction: 2=down, 3=up
        vert_dir = jnp.where(vert_dist_down < vert_dist_up, 2, 3)

        # Compare minimal distances and choose direction
        move_horiz = (min_horiz_dist > min_vert_dist)

        # Select actions based on comparison
        actions = jnp.where(move_horiz, horiz_dir, vert_dir)

        return actions

    def get_moves_away_from_coin(self, moves_toward: jnp.ndarray) -> jnp.ndarray:
        """
        Given a [n_agents] array of moves in {0,1,2,3} that would take each agent
        *toward* the coin, returns a [n_agents] array of moves that take the agents
        *away* from the coin. The mapping is:
           0->1, 1->0, 2->3, 3->2
        Because 0=right vs. 1=left, 2=down vs. 3=up.
        """
        # moves_toward, shape [n_agents], each in {0,1,2,3}
        moves_away = jnp.zeros_like(moves_toward)
        moves_away = jnp.where(moves_toward == 0, 1, moves_away)  # right->left
        moves_away = jnp.where(moves_toward == 1, 0, moves_away)  # left->right
        moves_away = jnp.where(moves_toward == 2, 3, moves_away)  # down->up
        moves_away = jnp.where(moves_toward == 3, 2, moves_away)  # up->down
        return moves_away

    def get_coop_actions(self, state: CoinGameState) -> jnp.ndarray:
        """
        Returns a [n_agents] array of "cooperative" actions:
          - The coin's owner (coin_color) moves TOWARD the coin.
          - Every other agent moves AWAY from the coin.
        """
        moves_toward = self.get_moves_towards_coin(state)  # shape [n_agents]
        moves_away   = self.get_moves_away_from_coin(moves_toward)

        # For the agent i = coin_color, we pick moves_toward[i].
        # For i != coin_color, pick moves_away[i].
        agent_indices = jnp.arange(self.n_agents, dtype=jnp.int32)
        coin_color = state.coin_color  # an integer in [0..n_agents)

        cooperative_actions = jnp.where(
            agent_indices == coin_color,
            moves_toward,  # if i == coin_color
            moves_away     # else
        )
        return cooperative_actions
