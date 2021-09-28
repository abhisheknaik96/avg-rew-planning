"""Tests for basic functioning of DVI algorithms."""
import functools
import itertools
from typing import Callable

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import dvi
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import structure


# Brute force - can stop using
def calc_stationary_distribution(transitions):
  # Subtract eye from transitions
  transitions_minus_I = transitions - np.eye(len(transitions))
  bottom_row = np.ones(len(transitions), dtype=transitions.dtype)
  augmented_transitions = np.vstack((transitions_minus_I, bottom_row))
  solution = np.zeros(len(transitions)+1, dtype=transitions.dtype)
  solution[-1] = 1.
  return np.linalg.solve(
      augmented_transitions.T.dot(augmented_transitions),
      augmented_transitions.T.dot(solution))

_MAKE_DVI = functools.partial(dvi.Evaluation,         step_size=.5,
                              beta=.5,
                              initial_r_bar=.5,
                              synchronized=True,
                              )

class DVIEvaluationTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.float32, _MAKE_DVI,micro.create_mrp1, (-1/3, 0, 1/3)),
                            )
  def test_correct_values(self, dtype: np.dtype, alg_constructor, env_constructor, want_values):
    tolerance_places = 6 if dtype is np.float32 else 10
    # environment = micro.create_mrp1(dtype)
    environment = env_constructor(dtype)
    algorithm = alg_constructor(
        mrp=environment,
        initial_values=np.zeros(environment.num_states, dtype=dtype))
    # algorithm = dvi.Evaluation(
    #     mrp=environment,
    #     step_size=.5,
    #     beta=.5,
    #     initial_r_bar=.5,
    #     initial_values=np.zeros(environment.num_states, dtype=dtype),
    #     synchronized=True)

    for _ in range(50):
      changes = algorithm.update()

    values = algorithm.get_estimates()['v']
    # print(f'converged to: {values}')
    stationary_dist = calc_stationary_distribution(environment.transitions)
    print(f'stationary_dist is:', stationary_dist)
    mc = environment.as_markov_chain()
    print(f'quantecon says: {mc.stationary_distributions}')

    offset = stationary_dist.dot(algorithm.get_estimates()['v'])
    # print('offset is: ', offset)
    centered_values = values - offset
    print(f'centered values: {centered_values}')
    with self.subTest('correct_centered_values'):
      np.testing.assert_array_almost_equal(centered_values, want_values)

if __name__ == '__main__':
  absltest.main()
