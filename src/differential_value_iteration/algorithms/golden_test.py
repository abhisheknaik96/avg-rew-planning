"""Tests that prediction algorithms learn correct values on sample MRPs."""
import functools
import itertools

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import dvi
from differential_value_iteration.algorithms import mdvi
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import micro

_MAKE_DVI = functools.partial(dvi.Evaluation,
                              step_size=.5,
                              beta=.5,
                              initial_r_bar=.0,
                              synchronized=True,
                              )
_MAKE_MDVI = functools.partial(mdvi.Evaluation,
                               step_size=.1,
                               beta=.1,
                               initial_r_bar=0.,
                               synchronized=True,
                               )
_MAKE_RVI = functools.partial(rvi.Evaluation,
                              step_size=.5,
                              reference_index=0,
                              synchronized=True,
                              )

_ALGS = (_MAKE_DVI, _MAKE_MDVI, _MAKE_RVI)
_MRP1 = {'constructor': micro.create_mrp1, 'values': ((-1 / 3, 0, 1 / 3),)}
_MRP2 = {'constructor': micro.create_mrp2, 'values': (
(-730 / 273, 20 / 273, 710 / 273, np.Inf), (np.Inf, np.Inf, np.Inf, 0.))}
_MRP3 = {'constructor': micro.create_mrp3, 'values': ((0., 0.),)}

_MRPS = (_MRP1, _MRP2, _MRP3)
_DTYPES = (np.float32, np.float64)


class EvaluationGoldenTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(_DTYPES, _ALGS, _MRPS))
  def test_correct_values_multichain(self, dtype: np.dtype, alg_constructor,
      env_dict):
    env_constructor = env_dict['constructor']
    tolerance_places = 4 if dtype is np.float32 else 10
    environment = env_constructor(dtype)
    algorithm = alg_constructor(
        mrp=environment,
        initial_values=np.zeros(environment.num_states, dtype=dtype))

    for _ in range(250):
      changes = algorithm.update()

    values = algorithm.get_estimates()['v']
    mc = environment.as_markov_chain()
    with self.subTest('correct stationary distributions'):
      self.assertEqual(len(mc.stationary_distributions),
                       len(env_dict['values']))

    # If the problem has multiple stationary distributions, iterate them.
    for (stationary_distribution, want_values) in zip(
        mc.stationary_distributions, env_dict['values']):
      offset = stationary_distribution.dot(algorithm.get_estimates()['v'])
      centered_values = values - offset

      # Clear out values for states not in distribution.
      centered_values = np.where(stationary_distribution == 0, 0.,
                                 centered_values)
      want_values = np.where(stationary_distribution == 0, 0., want_values)

      with self.subTest('correct_centered_values'):
        np.testing.assert_array_almost_equal(centered_values, want_values,
                                             decimal=tolerance_places)


if __name__ == '__main__':
  absltest.main()
