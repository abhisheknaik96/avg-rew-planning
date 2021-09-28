"""Tests some basic structure.py functionality."""
from typing import Sequence

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro

_MRPS = (micro.create_mrp1, micro.create_mrp2, micro.create_mrp3)
_MDPS = (micro.create_mdp1, micro.create_mdp2, micro.create_mdp3, garet.GARET1,
         garet.GARET2, garet.GARET3)


class StructureTest(parameterized.TestCase):

  @parameterized.parameters(*_MRPS)
  def test_create_mrp(self, mrp_constructor):
    mrp = mrp_constructor(dtype=np.float32)
    self.assertTrue(mrp is not None)

  @parameterized.parameters(
      (micro.create_mrp1, 1),
      (micro.create_mrp2, 2),
      (micro.create_mrp3, 1))
  def test_mrp_extract_markov_chain(self, mrp_constructor, num_classes):
    mrp = mrp_constructor(dtype=np.float32)
    mc = mrp.as_markov_chain()
    with self.subTest('markov_chain_success'):
      self.assertTrue(mc is not None)
    with self.subTest('num_communication_classes'):
      self.assertEqual(mc.num_communication_classes, num_classes)

  @parameterized.parameters(*_MDPS)
  def test_create_mdp(self, mdp_constructor):
    mdp = mdp_constructor(dtype=np.float32)
    self.assertTrue(mdp is not None)

  @parameterized.parameters(
      (micro.create_mdp1, (0, 0), 2),
      (micro.create_mdp1, (0, 1), 2),
      (micro.create_mdp1, (1, 0), 2),
      (micro.create_mdp1, (1, 1), 1),
      (micro.create_mdp2, (0, 0), 2),
      (micro.create_mdp2, (0, 1), 2),
      (micro.create_mdp2, (1, 0), 2),
      (micro.create_mdp2, (1, 1), 2),
      (micro.create_mdp3, (0, 0, 0), 1),
      (micro.create_mdp3, (0, 0, 1), 1),
      (micro.create_mdp3, (0, 1, 0), 1),
      (micro.create_mdp3, (0, 1, 1), 1),
      (micro.create_mdp3, (1, 0, 0), 1),
      (micro.create_mdp3, (1, 0, 1), 1),
      (micro.create_mdp3, (1, 1, 0), 1),
      (micro.create_mdp3, (1, 1, 1), 1),
  )
  def test_mdp_extract_markov_chain_deterministic_policy(self,
      mdp_constructor,
      policy: Sequence[int],
      num_classes: int):
    mdp = mdp_constructor(dtype=np.float32)
    mc = mdp.as_markov_chain_from_deterministic_policy(policy)
    with self.subTest('markov_chain_success'):
      self.assertTrue(mc is not None)
    with self.subTest('num_communication_classes'):
      self.assertEqual(mc.num_communication_classes, num_classes)

  @parameterized.parameters(
      (micro.create_mdp1, ((1., 1.), (0., 0.)), 2),
      (micro.create_mdp1, ((.5, .5), (.5, .5)), 1),
      (micro.create_mdp2, ((.5, .5), (.5, .5)), 2),
      (micro.create_mdp3, ((1., 1., 1.), (0., 0., 0.)), 1),
      (micro.create_mdp3, ((.5, .5, .5), (.5, .5, .5)), 1),
  )
  def test_mdp_extract_markov_chain_stochastic_policy(self,
      mdp_constructor,
      policy: Sequence[Sequence[float]],
      num_classes: int):
    policy = np.array(policy)
    mdp = mdp_constructor(dtype=np.float32)
    mc = mdp.as_markov_chain_from_stochastic_policy(policy)
    with self.subTest('markov_chain_success'):
      self.assertTrue(mc is not None)
    with self.subTest('num_communication_classes'):
      self.assertEqual(mc.num_communication_classes, num_classes)

  @parameterized.parameters(
      (micro.create_mdp1, 1),
      (micro.create_mdp2, 2),
      (micro.create_mdp3, 1),
  )
  def test_mdp_extract_markov_chain_equiprob_policy(self,
      mdp_constructor,
      num_classes: int):
    mdp = mdp_constructor(dtype=np.float32)
    mc = mdp.as_markov_chain()
    with self.subTest('markov_chain_success'):
      self.assertTrue(mc is not None)
    with self.subTest('num_communication_classes'):
      self.assertEqual(mc.num_communication_classes, num_classes)

if __name__ == '__main__':
  absltest.main()
