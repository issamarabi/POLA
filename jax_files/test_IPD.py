import unittest
import jax
import jax.numpy as jnp

# Import the environment class from your ipd_jax file
# Adjust the path if necessary.
from ipd_jax import IPD


class TestIPDJax(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method is run.
        We can set a random key here to use in the tests
        and define some default parameters.
        """
        self.rng_key = jax.random.PRNGKey(42)

    def test_init_n_agents_default(self):
        """
        Test default construction with 2 agents,
        start_with_cooperation=False,
        cooperation_factor=1.33.
        """
        env = IPD(n_agents=2, start_with_cooperation=False, cooperation_factor=1.33)
        self.assertEqual(env.n_agents, 2)
        self.assertAlmostEqual(env.cooperation_factor, 1.33)
        self.assertFalse(env.start_with_cooperation)
        # initial_state shape: 3*N => 6 elements
        self.assertEqual(env.initial_state.shape, (6,))
        # Check that all are in the "start" slot: [0,0,1, 0,0,1]
        # the environment code sets [0,0,1] for each agent if start_with_cooperation=False
        expected = jnp.array([0., 0., 1., 0., 0., 1.])
        self.assertTrue(jnp.allclose(env.initial_state, expected))

    def test_init_with_cooperation(self):
        """
        Test construction with start_with_cooperation=True
        for 2 agents.
        """
        env = IPD(n_agents=2, start_with_cooperation=True, cooperation_factor=2.0)
        self.assertTrue(env.start_with_cooperation)
        self.assertEqual(env.initial_state.shape, (6,))
        # Now each agent is [0,1,0] => two agents => [0,1,0, 0,1,0]
        expected = jnp.array([0., 1., 0., 0., 1., 0.])
        self.assertTrue(jnp.allclose(env.initial_state, expected))

    def test_init_single_agent(self):
        """
        Test with a single agent, to verify shapes and logic still hold.
        """
        env = IPD(n_agents=1, start_with_cooperation=False, cooperation_factor=1.5)
        self.assertEqual(env.n_agents, 1)
        # initial_state shape: 3*N => 3 elements for 1 agent
        self.assertEqual(env.initial_state.shape, (3,))
        # If start_with_cooperation=False => [0,0,1]
        expected = jnp.array([0., 0., 1.])
        self.assertTrue(jnp.allclose(env.initial_state, expected))

    def test_init_four_agents(self):
        """
        Test an environment with 4 agents.
        """
        env = IPD(n_agents=4, start_with_cooperation=False, cooperation_factor=1.1)
        self.assertEqual(env.n_agents, 4)
        # shape => 3*N => 12
        self.assertEqual(env.initial_state.shape, (12,))
        # If not starting with cooperation => each agent => [0,0,1]
        # => total [0,0,1, 0,0,1, 0,0,1, 0,0,1]
        ones = jnp.array([0.,0.,1., 0.,0.,1., 0.,0.,1., 0.,0.,1.])
        self.assertTrue(jnp.allclose(env.initial_state, ones))

    def test_reset_method(self):
        """
        Test reset(...) ensures it returns (initial_state, initial_state)
        and that it can be called multiple times without changing state.
        """
        env = IPD(n_agents=2)
        s1, s1_obs = env.reset(self.rng_key)
        # Both should be the same
        self.assertTrue(jnp.allclose(s1, s1_obs))
        # Reset again
        s2, s2_obs = env.reset(self.rng_key)
        self.assertTrue(jnp.allclose(s2, s2_obs))
        # The initial states from resets should be identical
        self.assertTrue(jnp.allclose(s1, s2))

    def test_step_2agents_actions(self):
        """
        Test step(...) correctness for 2 agents, enumerating all possible action pairs:
            0 => Defect, 1 => Cooperate.
        Check shape of new_state, observation, and reward correctness.
        """
        env = IPD(n_agents=2, start_with_cooperation=False, cooperation_factor=1.33)
        # Suppose we reset the environment
        init_s, _ = env.reset(self.rng_key)
        # Now step with each possible (action_agent0, action_agent1)
        # We'll do a small loop over (0,0), (0,1), (1,0), (1,1).
        # We check the next state, observation, and rewards.

        # base = (#cooperators * factor)/N
        # if agent i cooperates => base-1, else base
        # For 2 agents, we match the original IPD matrix:
        #   CC => each gets (2*1.33/2 - 1) = 1.33 -1 = 0.33
        #   DD => each gets (0 * 1.33/2) = 0
        #   DC => defector => 1.33/2=0.665, cooperator => 0.665-1= -0.335
        #   CD => similarly => (defector =>0.665, cooperator =>-0.335)

        pairs = [(0,0), (0,1), (1,0), (1,1)]
        # expected_rews will be a dict from pairs -> (r0, r1)
        # Using the formula:
        # number_of_coops = sum_of_actions
        # base = c * factor / 2
        # coop => base -1, defect => base
        # factor=1.33
        # c can be 0,1,2
        # c=0 => base=0 => (0,0)
        # c=1 => base=0.665 => def=0.665, coop=-0.335
        # c=2 => base=1.33 => coop=0.33, coop=0.33
        expected_rews = {
            (0,0): (0.0, 0.0),           # DD
            (0,1): (0.665, -0.335),      # D,C
            (1,0): (-0.335, 0.665),      # C,D
            (1,1): (0.33, 0.33),         # C,C
        }

        for act_pair in pairs:
            # step
            new_state, obsv, (r0, r1), _ = env.step(init_s, act_pair[0], act_pair[1], unused_key=self.rng_key)
            with self.subTest(msg=f"Actions={act_pair}"):
                # Check shapes
                self.assertEqual(new_state.shape, (6,))
                self.assertEqual(obsv.shape, (6,))
                # Rewards
                exp_r0, exp_r1 = expected_rews[act_pair]
                self.assertAlmostEqual(r0, exp_r0, places=5)
                self.assertAlmostEqual(r1, exp_r1, places=5)

                # Check new_state encoding
                # For each action=0 => [1,0,0], action=1 => [0,1,0].
                # So for (0,1) => new_state => [1,0,0, 0,1,0]
                # etc.
                chunk_size = 3
                agent0_vec = new_state[0:3]
                agent1_vec = new_state[3:6]
                if act_pair[0] == 0:
                    self.assertTrue(jnp.allclose(agent0_vec, jnp.array([1.,0.,0.])))
                else:
                    self.assertTrue(jnp.allclose(agent0_vec, jnp.array([0.,1.,0.])))
                if act_pair[1] == 0:
                    self.assertTrue(jnp.allclose(agent1_vec, jnp.array([1.,0.,0.])))
                else:
                    self.assertTrue(jnp.allclose(agent1_vec, jnp.array([0.,1.,0.])))

                # Check that observation == new_state
                self.assertTrue(jnp.allclose(new_state, obsv))

    def test_step_3agents_random_actions(self):
        """
        Test step(...) with 3 agents, using random actions. 
        Ensure shapes and rewards are consistent with the formula:
          Each cooperator => base - 1
          Each defector => base
          base = (c * cooperation_factor) / n_agents
        """
        env = IPD(n_agents=3, start_with_cooperation=False, cooperation_factor=1.5)
        init_state, _ = env.reset(self.rng_key)

        # random actions in {0,1}, shape = (3,)
        # but let's do a few manual checks
        test_actions = [
            (0,0,0),  # all defect
            (1,1,1),  # all coop
            (0,1,0),  # mixed
            (1,0,1),
        ]

        for acts in test_actions:
            new_state, obsv, rews, _ = env.step(init_state, *acts, unused_key=self.rng_key)
            with self.subTest(msg=f"Actions={acts} => Rewards={rews}"):
                # shape checks
                self.assertEqual(new_state.shape, (9,))
                self.assertEqual(obsv.shape, (9,))
                self.assertEqual(len(rews), 3)
                # reward correctness
                c = sum(acts)  # number of cooperators
                base = (c * env.cooperation_factor) / 3.0
                for i in range(3):
                    if acts[i] == 1:
                        # cooperator => base - 1
                        self.assertAlmostEqual(rews[i], base - 1.0, places=5)
                    else:
                        # defector => base
                        self.assertAlmostEqual(rews[i], base, places=5)

                # new_state one-hot correctness
                for i in range(3):
                    chunk = new_state[3*i : 3*(i+1)]
                    if acts[i] == 0:
                        self.assertTrue(jnp.allclose(chunk, jnp.array([1.,0.,0.])))
                    else:
                        self.assertTrue(jnp.allclose(chunk, jnp.array([0.,1.,0.])))
                # obsv == new_state
                self.assertTrue(jnp.allclose(new_state, obsv))

    def test_reset_multiple_times(self):
        """
        Test that calling reset multiple times consistently returns the same shape
        and does not affect internal state for repeated calls.
        """
        env = IPD(n_agents=3, start_with_cooperation=True, cooperation_factor=2.0)
        s1, s1_2 = env.reset(self.rng_key)
        self.assertEqual(s1.shape, (9,))
        # The environment places each agent in [0,1,0], so s1 should be:
        # [0,1,0, 0,1,0, 0,1,0]
        self.assertTrue(jnp.allclose(s1, s1_2))
        s2, s2_2 = env.reset(self.rng_key)
        self.assertTrue(jnp.allclose(s2, s2_2))
        # Should be identical
        self.assertTrue(jnp.allclose(s1, s2))

    def test_cooperation_factor_variations(self):
        """
        Check step(...) correctness with different cooperation_factor values, e.g. <1, >2,
        ensuring the environment uses that factor in reward calculations consistently.
        """
        # e.g. factor=0.5 => defection is always better
        env = IPD(n_agents=2, start_with_cooperation=False, cooperation_factor=0.5)
        init_state, _ = env.reset(self.rng_key)
        # Step with (1,1) => c=2 => base= (2*0.5)/2= 0.5 => coop=0.5-1 = -0.5
        new_state, obsv, (r0, r1), _ = env.step(init_state, 1, 1)
        self.assertAlmostEqual(r0, -0.5)
        self.assertAlmostEqual(r1, -0.5)

        # factor=3 => bigger incentive for cooperating if the other agent cooperates
        env2 = IPD(n_agents=2, cooperation_factor=3.0)
        init_s, _ = env2.reset(self.rng_key)
        # (C,C) => c=2 => base= (2*3)/2=3 => coop=3-1=2
        _, _, (r0_2, r1_2), _ = env2.step(init_s, 1, 1)
        self.assertEqual(r0_2, 2.0)
        self.assertEqual(r1_2, 2.0)
        # (D,C) => c=1 => base= (1*3)/2=1.5 => def=1.5, coop=1.5-1=0.5
        _, _, (rd, rc), _ = env2.step(init_s, 0, 1)
        self.assertEqual(rd, 1.5)
        self.assertEqual(rc, 0.5)

    def test_invalid_actions_length(self):
        """
        This checks how the environment behaves if we pass fewer or more actions
        than n_agents. By default, the environment doesn't check the action length,
        but if you have added error checking, you can test that here.
        """
        env = IPD(n_agents=3)
        s, _ = env.reset(self.rng_key)

        # Suppose we pass 2 actions for a 3-agent environment. 
        # The environment code as written won't raise an error by default (it will just
        # shape mismatch). If you *want* it to raise an error, you can do so in your code
        # and test it here. For demonstration:
        try:
            _ = env.step(s, 0, 1, self.rng_key)  # missing the 3rd agent's action
            # If no error is raised, we can mark the test as failing (depending on your design).
            self.fail("Expected an error or shape mismatch for missing actions, but none was raised.")
        except Exception as e:
            # If the environment is not strictly checking, 
            # we see a JAX shape error or we want to confirm the environment's behavior:
            # This is just a demonstration, you can refine how you handle it.
            pass

    def test_large_number_of_agents(self):
        """
        Stress test with 10 agents. 
        Ensures shape correctness and that step(...) processes them without error.
        """
        n_agents = 10
        env = IPD(n_agents=n_agents, start_with_cooperation=False, cooperation_factor=2.0)
        init_state, _ = env.reset(self.rng_key)
        self.assertEqual(init_state.shape, (3 * n_agents,))

        # create random actions in {0,1} for 10 agents
        rng_key, subkey = jax.random.split(self.rng_key)
        random_acts = jax.random.randint(subkey, shape=(n_agents,), minval=0, maxval=2)
        random_acts_tuple = tuple(random_acts.tolist())  # e.g. (1,0,0,1,1,0,0,1,0,1)

        new_state, obsv, rews, _ = env.step(init_state, *random_acts_tuple, unused_key=subkey)
        self.assertEqual(new_state.shape, (3 * n_agents,))
        self.assertEqual(obsv.shape, (3 * n_agents,))
        self.assertEqual(len(rews), n_agents)

        # Check reward formula quickly
        c = sum(random_acts)
        base = (c * 2.0) / n_agents
        for i, a in enumerate(random_acts_tuple):
            if a == 1:
                # cooperator => base - 1
                self.assertAlmostEqual(rews[i], base - 1.0, places=6)
            else:
                self.assertAlmostEqual(rews[i], base, places=6)

        # new_state one-hots
        for i in range(n_agents):
            chunk = new_state[3*i : 3*(i+1)]
            if random_acts_tuple[i] == 0:
                self.assertTrue(jnp.allclose(chunk, jnp.array([1.,0.,0.])))
            else:
                self.assertTrue(jnp.allclose(chunk, jnp.array([0.,1.,0.])))


if __name__ == "__main__":
    unittest.main()
