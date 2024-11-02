import jax.numpy as jnp

class IPD:
    """
    A generalized N-agent vectorized environment for the Iterated Prisoner's Dilemma (IPD).

    Each agent can take one of two actions:
        0 -> Defect (D)
        1 -> Cooperate (C)

    The environment state is a (3*N)-dimensional one-hot vector encoding the
    last action of each agent as [1,0,0] = Defect, [0,1,0] = Cooperate, or [0,0,1] = Start.
    For example, if N=2 and both agents cooperated last step, the state would be:
        [0,1,0, 0,1,0].
    If N=3 and agent0=Coop, agent1=Defect, agent2=Start, the state would be:
        [0,1,0, 1,0,0, 0,0,1], and so on.

    Rewards follow a generalized IPD payoff:
        - Let c = number of cooperators among N agents at this step.
        - Each cooperator gets (c * cooperation_factor / N) - 1
        - Each defector gets (c * cooperation_factor / N).

    If you set N=2, this reproduces the same payoff structure and learning
    dynamics of the original 2-agent IPD code, but in a more flexible form.
    """

    def __init__(
        self,
        n_agents=2,
        start_with_cooperation=False,
        cooperation_factor=1.33
    ):
        """
        Initialize the IPD environment.

        Args:
            n_agents: Number of agents in the game.
            start_with_cooperation: Whether to start each agent's one-hot as [0,1,0] (cooperate)
                                    instead of [0,0,1] (start).
            cooperation_factor: Factor determining the reward for cooperation relative to defection.
        """
        self.n_agents = n_agents
        self.cooperation_factor = cooperation_factor
        self.start_with_cooperation = start_with_cooperation

        # Build the initial state.
        # If 'start_with_cooperation' is True, each agent's initial one-hot is [0,1,0].
        # Otherwise, each agent's initial one-hot is [0,0,1] (the "start" slot).
        onehots = []
        if self.start_with_cooperation:
            # All agents start as if they last "cooperated"
            for _ in range(self.n_agents):
                onehots.append(jnp.array([0., 1., 0.]))
        else:
            # All agents start in the "start" slot
            for _ in range(self.n_agents):
                onehots.append(jnp.array([0., 0., 1.]))

        # Concatenate to get a (3*N,)-dim vector
        self.initial_state = jnp.concatenate(onehots, axis=0)

    def reset(self, unused_key):
        """
        Reset the game to its initial state.

        Only first element needed, but we keep this signature to avoid breaking existing code.
        """
        return self.initial_state, self.initial_state

    def step(self, unused_state, *actions, unused_key=None):
        """
        Execute a step in the game based on the actions of N agents.

        Args:
            unused_state: Unused in IPD, since our "state" is fully captured by last step's actions.
            *actions: A variable-length tuple of length N, with each action in {0,1} = (D, C).
            unused_key: A placeholder PRNG key (for vectorized usage).

        Returns:
            (new_state, observation, rewards, None)

            where:
              - new_state:  (3*N,)-dim one-hot vector for each agent's new action.
              - observation: identical to new_state (each agent sees the full last-action record).
              - rewards: a tuple of length N, containing each agent's reward.
              - None: placeholder for extra info (unused).
        """

        # 1) Convert the *actions tuple into an array of shape (N,).
        actions_array = jnp.array(actions, dtype=jnp.int32)  # shape = (N,)

        # 2) Count the number of cooperators.
        c = jnp.sum(actions_array)  # shape=(), sum of 0/1 actions

        # 3) Compute each agent's reward with the formula:
        #    base = (c * cooperation_factor) / N
        #    if agent i cooperates => reward_i = base - 1
        #    if agent i defects   => reward_i = base
        base = (c * self.cooperation_factor) / float(self.n_agents)

        def per_agent_reward(a):
            # a=1 => cooperator => base -1
            # a=0 => defector   => base
            return jnp.where(a == 1, base - 1.0, base)

        rewards_array = jnp.vectorize(per_agent_reward)(actions_array)
        rewards_tuple = tuple(rewards_array)  # e.g. (r1, r2, ..., rN)

        # 4) Construct the new (3*N,)-dim state vector by one-hotting each agent's last action
        #    Defect=0 => [1,0,0], Cooperate=1 => [0,1,0]. We never reuse the "start" slot after the first step.
        new_onehots = []
        for a in actions_array:
            if a == 0:
                new_onehots.append(jnp.array([1., 0., 0.]))  # Defect
            else:
                new_onehots.append(jnp.array([0., 1., 0.]))  # Cooperate

        new_state = jnp.concatenate(new_onehots, axis=0)

        # 5) In this setup, each agent sees the same "observation", i.e. the global last-action vector.
        observation = new_state

        return new_state, observation, rewards_tuple, None
