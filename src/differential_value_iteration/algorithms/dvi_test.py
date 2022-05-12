"""Tests for basic functioning of DVI algorithms."""
import itertools
from typing import Callable

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import dvi
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import structure

_GARET1, _GARET2, _GARET3 = garet.GARET1, garet.GARET2, garet.GARET3
_DTYPES = (np.float64, )

class DVIEvaluationTest(parameterized.TestCase):

  @parameterized.parameters(_DTYPES,)
  def test_dvi_sync_converges(self, dtype: np.dtype):
    tolerance_places = 6 if dtype is np.float32 else 10
    environment = micro.create_mrp1(dtype)
    algorithm = dvi.Evaluation(
        mrp=environment,
        step_size=.5,
        beta=.5,
        divide_stepsizes_by_num_states=True,
        initial_r_bar=.5,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)

    for _ in range(50):
      changes = algorithm.update()

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(np.sum(np.abs(changes)), 0.,
                             places=tolerance_places)

  @parameterized.parameters(_DTYPES)
  def test_dvi_sync_converges(self, dtype: np.dtype):
    tolerance_places = 6 if dtype is np.float32 else 10
    environment = micro.create_mrp1(dtype)
    algorithm = dvi.Evaluation(
        mrp=environment,
        step_size=1.,
        beta=1.,
        divide_stepsizes_by_num_states=True,
        initial_r_bar=.5,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)

    for _ in range(50):
      changes = algorithm.update()

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(np.sum(np.abs(changes)), 0.,
                             places=tolerance_places)

  @parameterized.parameters(_DTYPES)
  def test_dvi_async_converges(self, dtype: np.dtype):
    tolerance_places = 6 if dtype is np.float32 else 10
    environment = micro.create_mrp1(dtype)
    algorithm = dvi.Evaluation(
        mrp=environment,
        step_size=1.,
        beta=1.,
        divide_stepsizes_by_num_states=True,
        initial_r_bar=.5,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=False)

    for _ in range(50):
      change_sum = 0.
      for _ in range(environment.num_states):
        change = algorithm.update()
        change_sum += np.abs(change)

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(change_sum, 0., places=tolerance_places)


class DVIControlTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(
      (micro.create_mdp1, _GARET1, _GARET2, _GARET3), _DTYPES)
  )
  def test_dvi_sync_converges(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      dtype: np.dtype):
    tolerance_places = 10 if dtype == np.float64 else 5
    environment = mdp_constructor(dtype=dtype)
    algorithm = dvi.Control(
        mdp=environment,
        step_size=1.,
        beta=1.,
        divide_stepsizes_by_num_states=True,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)

    for _ in range(500):
      changes = algorithm.update()

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(np.max(np.abs(changes)), 0.,
                             places=tolerance_places)

  @parameterized.parameters(itertools.product(
      (micro.create_mdp1, _GARET1, _GARET2, _GARET3), _DTYPES)
  )
  def test_dvi_async_converges(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      dtype: np.dtype):
    tolerance_places = 9 if dtype == np.float64 else 6
    environment = mdp_constructor(dtype=dtype)
    algorithm = dvi.Control(
        mdp=environment,
        step_size=.5,
        beta=.5,
        divide_stepsizes_by_num_states=True,
        initial_r_bar=.0,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=False)

    for _ in range(750):
      change_max = 0.
      for _ in range(environment.num_states):
        change = algorithm.update()
        change_max = np.maximum(change_max, np.abs(change))

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(change_max, 0., places=tolerance_places)


if __name__ == '__main__':
  absltest.main()
