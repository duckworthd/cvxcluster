from time import time
import os

import numpy as np

from cvxcluster import *


class SolverTests(object):

  def _load(self, fname):
    folder = os.path.split(__file__)[0]
    path   = os.path.join(folder, fname)
    return np.load(path)

  def load(self, folder, regularizer):
    if regularizer == "l2":
      class P(Problem, L2): pass
    if regularizer == "inf":
      class P(Problem, L_Inf): pass
    if regularizer == "l1":
      class P(Problem, L1): pass

    data     = self._load(os.path.join(folder, "problem.npz"))
    problem  = P(data['X'], data['gamma'], data['w'])

    data     = self._load(os.path.join(folder, "solution_{}.npz".format(regularizer)))
    solution = Solution(problem, data['lmbd'])

    return problem, solution

  def solve(self, problem, solution, maxtime=30):
    t0 = time()
    primal, dual, gap = solution.duality_gap()
    for solution_ in self.solver.minimize(problem):
      primal_, dual_, gap_ = solution_.duality_gap()
      if gap_ <= gap: return
      self.assertGreater(maxtime, time() - t0)

  def test_simple_l1(self):
    problem, solution = self.load("simple", "l1")
    self.solve(problem, solution)

  def test_simple_l2(self):
    problem, solution = self.load("simple", "l2")
    self.solve(problem, solution)

  def test_simple_inf(self):
    problem, solution = self.load("simple", "inf")
    self.solve(problem, solution)
