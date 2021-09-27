"""Tests that our control algorithms find same policy on test problems."""
import itertools
from typing import Callable

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import dvi
from differential_value_iteration.algorithms import mdvi
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import structure

_MDPS = (micro.create_mdp1, micro.create_mdp2, garet.GARET1, garet.GARET2,
         garet.GARET3)


class PolicyTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(_MDPS, (np.float32, np.float64)))
  def test_identical_policies_sync(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      dtype: np.dtype):
    environment = mdp_constructor(dtype=dtype)
    rvi_control = rvi.Control(
        mdp=environment,
        step_size=.75,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        reference_index=0,
        synchronized=True)
    dvi_control = dvi.Control(
        mdp=environment,
        step_size=.1,
        beta=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)
    mdvi_control_1 = mdvi.Control1(
        mdp=environment,
        step_size=.1,
        beta=.1,
        threshold=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)
    mdvi_control_2 = mdvi.Control2(
        mdp=environment,
        step_size=.1,
        beta=.1,
        threshold=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)
    for i in range(500):
      rvi_control.update()
      dvi_control.update()
      mdvi_control_1.update()
      mdvi_control_2.update()
    with self.subTest('rvi vs dvi'):
      np.testing.assert_array_equal(rvi_control.greedy_policy(),
                                    dvi_control.greedy_policy())
    with self.subTest('rvi vs mdvi1'):
      np.testing.assert_array_equal(rvi_control.greedy_policy(),
                                    mdvi_control_1.greedy_policy())
    with self.subTest('mdvi1 vs mdvi2'):
      np.testing.assert_array_equal(mdvi_control_1.greedy_policy(),
                                    mdvi_control_2.greedy_policy())

  @parameterized.parameters(itertools.product(_MDPS, (np.float32, np.float64)))
  def test_identical_policies_async(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      dtype: np.dtype):
    environment = mdp_constructor(dtype=dtype)
    rvi_control = rvi.Control(
        mdp=environment,
        step_size=.75,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        reference_index=0,
        synchronized=False)
    dvi_control = dvi.Control(
        mdp=environment,
        step_size=.1,
        beta=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=False)
    mdvi_control_1 = mdvi.Control1(
        mdp=environment,
        step_size=.1,
        beta=.1,
        threshold=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=False)
    mdvi_control_2 = mdvi.Control2(
        mdp=environment,
        step_size=.1,
        beta=.1,
        threshold=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=False)
    for _ in range(500):
      for _ in range(environment.num_states):
        rvi_control.update()
        dvi_control.update()
        mdvi_control_1.update()
        mdvi_control_2.update()
    with self.subTest('rvi vs dvi'):
      np.testing.assert_array_equal(rvi_control.greedy_policy(),
                                    dvi_control.greedy_policy())
    with self.subTest('rvi vs mdvi1'):
      np.testing.assert_array_equal(rvi_control.greedy_policy(),
                                    mdvi_control_1.greedy_policy())
    with self.subTest('mdvi1 vs mdvi2'):
      np.testing.assert_array_equal(mdvi_control_1.greedy_policy(),
                                    mdvi_control_2.greedy_policy())

  @parameterized.parameters(itertools.product(
      (garet.GARET1,),
      (np.float32,)))
  def test_identical_policy_values_sync(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      dtype: np.dtype):
    """Not useful now, can be used to compare return from different policies."""
    environment = mdp_constructor(dtype=dtype)
    mdvi_control_1 = mdvi.Control1(
        mdp=environment,
        step_size=.1,
        beta=.1,
        threshold=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)
    mdvi_control_2 = mdvi.Control2(
        mdp=environment,
        step_size=.1,
        beta=.1,
        threshold=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)
    for i in range(500):
      mdvi_control_1.update()
      mdvi_control_2.update()

    control_1_policy = mdvi_control_1.greedy_policy()
    control_2_policy = mdvi_control_2.greedy_policy()
    differences = np.sum(np.where(control_1_policy == control_2_policy, 0, 1))
    with self.subTest('policies match'):
      self.assertEqual(differences, 0)

    # Tuples are better for array indexing.
    control_1_policy = tuple(control_1_policy)
    control_2_policy = tuple(control_2_policy)

    initial_state_distribution = np.zeros(environment.num_states, np.float32)
    # Start from State 1
    initial_state_distribution[0] = 1.
    iterations = 100
    control_1_return = generate_rewards(environment,
                                        control_1_policy,
                                        iterations=iterations,
                                        initial_state_distribution=initial_state_distribution)
    control_2_return = generate_rewards(environment,
                                        control_2_policy,
                                        iterations=iterations,
                                        initial_state_distribution=initial_state_distribution)
    with self.subTest('returns match'):
      self.assertAlmostEqual(control_1_return, control_2_return)


def generate_rewards(environment: structure.MarkovDecisionProcess,
    policy: np.ndarray, iterations: int,
    initial_state_distribution: np.ndarray):
  total_return = 0.
  policy_rewards = environment.rewards[
    policy, np.arange(0, environment.num_states)]
  policy_transitions = environment.transitions[policy, np.arange(0, environment.num_states)]
  state_distribution = initial_state_distribution.copy()
  for _ in range(iterations):
    total_return += np.dot(policy_rewards, state_distribution)
    state_distribution = np.dot(policy_transitions, state_distribution)
  return total_return / iterations


if __name__ == '__main__':
  absltest.main()
