import jax.numpy as jnp

class IteratedPrisonersDilemma:
    """
    A two-agent vectorized environment for the Iterated Prisoner's Dilemma (IPD).
    Possible actions for each agent are (C)ooperate and (D)efect.
    
    Attributes:
    - reward_matrix: Matrix representing the rewards for each combination of actions.
    - state_representations: One-hot encoded representation of possible states.
    - initial_state: Initial state of the game.
    """
    
    def __init__(self, start_with_cooperation=False, cooperation_factor=1.33):
        """
        Initialize the IPD environment.
        
        Args:
        - start_with_cooperation: If True, the initial state is set to cooperation for both agents.
        - cooperation_factor: Factor determining the reward for cooperation relative to defection.
        """
        reward_coop_coop = cooperation_factor - 1.
        reward_defect_defect = 0.
        reward_defect_coop = cooperation_factor / 2.  # Reward when I defect and opponent cooperates.
        reward_coop_defect = cooperation_factor / 2. - 1  # Reward when I cooperate and opponent defects.
        
        # Define the reward matrix for the game.
        self.reward_matrix = jnp.array([[reward_defect_defect, reward_defect_coop], 
                                        [reward_coop_defect, reward_coop_coop]])
        
        # One-hot encoded representation of possible states.
        self.state_representations = jnp.array([[[1, 0, 0, 1, 0, 0],  # Defect-Defect
                                                 [1, 0, 0, 0, 1, 0]],  # Defect-Cooperate
                                                [[0, 1, 0, 1, 0, 0],  # Cooperate-Defect
                                                 [0, 1, 0, 0, 1, 0]]])  # Cooperate-Cooperate
        
        # Set the initial state.
        if start_with_cooperation:
            self.initial_state = jnp.array([0, 1, 0, 0, 1, 0])
        else:
            self.initial_state = jnp.array([0, 0, 1, 0, 0, 1])

    def reset(self, unused_key):
        """
        Reset the game to its initial state.
        
        Args:
        - unused_key: Placeholder for a random seed (not used in this method).
        
        Returns:
        - Tuple of initial states for both agents.
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
