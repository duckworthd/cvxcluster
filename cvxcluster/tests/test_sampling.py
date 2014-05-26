import numpy as np

from numpy.testing import assert_allclose

import unittest

from cvxcluster import *


class NormalSamplerTests(unittest.TestCase):

  def setUp(self):
    self.r       = np.random.RandomState(0)
    self.k       = 2
    self.dim     = 2
    self.sampler = NormalSampler(self.r, self.k, self.dim)

  def test_samplepoints(self):
    self.sampler.sample_points(100)


def test_nearest_neighbors():
  X = np.asarray([
    [0,0],
    [0,1],
    [1,0],

    [5,5],
    [5,6],
    [6,5],
  ])

  weights = nearest_neighbors(X, k=2, phi=1.0)

  assert len(weights) == 6
