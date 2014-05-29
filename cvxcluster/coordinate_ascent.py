from select import select
import sys

import numpy as np

from ._coordinate_ascent import coordinate_ascent_iteration
from .problem import Solution
from .profile import profile


__all__ = [
  'CoordinateAscent',
]

class CoordinateAscent(object):
  """Solve Convex Clustering with Coordinate Ascent on the dual"""

  @profile
  def minimize(self, problem, lmbd=None):
    X, gamma, w           = problem.X, problem.gamma, problem.w
    n_samples, n_features = problem.n_samples, problem.n_features
    n_pairs               = problem.n_pairs
    r                     = np.random.RandomState(0)
    p                     = problem._norm

    if lmbd is None:
      lmbd = np.zeros( (n_pairs, n_features) )

    while True:

      # for each l
      #   lmbd_{l}^{t+1} = -0.5 * [
      #       (Delta_{l_1} - lmbd_{l}^{t}) - (Delta_{l_2} - lmbd_{l}^{t})
      #       + x_{l_1} - x_{l_2}
      #   ]
      #   lmbd[l]^{t+1} = project lmbd_l^{t+1} onto ||.||_{*} <= w_l * gamma

      Delta = problem.Delta(lmbd)

      yield Solution(problem, lmbd=lmbd, Delta=Delta)

      # use Cython implementation of coordinate ascent It gives ~2x gain over a
      # vanilla Python implementation
      coordinate_ascent_iteration(X, Delta, w, lmbd, gamma, p)
