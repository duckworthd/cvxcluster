import itertools
import unittest

from numpy.testing import assert_allclose
import numpy as np

from cvxcluster import *
from cvxcluster.tests import SolverTests
from cvxcluster.utils import *


class SolutionTests(unittest.TestCase):

  def setUp(self):
    problem, solution = SolverTests().load("simple", "l2")
    self.problem      = problem
    self.solution     = solution

  def test_v(self):
    v = self.solution.v
    u = self.solution.u
    for l, (i, j, _) in enumerate(iterrows(self.problem.w)):
      assert_allclose(v[l], u[i] - u[j])

  def test_clusters(self):
    v = np.ones(self.solution.lmbd.shape)

    # [0, 1, 2]
    for l, (i, j, _) in enumerate(iterrows(self.problem.w)):
      if i in [0,1,2] and j in [0,1,2]:
        v[l] = np.zeros(len(v[l]))
      if i in [4,5] and j in [4,5]:
        v[l] = np.zeros(len(v[l]))

    self.solution._v = v
    labels = self.solution.clusters

    self.assertEqual([0,0,0,1,2,2], list(labels))
