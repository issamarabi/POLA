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

    def step(self, unused_state, action_agent1, action_agent2, unused_key):
        """
        Execute a step in the game based on the actions of the two agents.
        
        Args:
        - unused_state: Placeholder for the current state (not used in this method).
        - action_agent1: Action taken by the first agent.
        - action_agent2: Action taken by the second agent.
        - unused_key: Placeholder for a random seed (not used in this method).
        
        Returns:
        - Tuple containing the new state, observation, and rewards for both agents.
        """
        reward_agent1 = self.reward_matrix[action_agent1, action_agent2]
        reward_agent2 = self.reward_matrix[action_agent2, action_agent1]
        new_state = self.state_representations[action_agent1, action_agent2]
        observation = new_state
        rewards = (reward_agent1, reward_agent2)
        
        return new_state, observation, rewards, None
