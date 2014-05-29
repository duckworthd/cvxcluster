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

  def __init__(self):
    # what's going on here, you ask? I'm forcing to compile JIT code now by
    # calling it once. The reason is, JIT compilation skews profiling times if
    # it needs to happen on the first iteration.
    X = np.zeros( (2,3), dtype=float)
    coordinate_ascent_iteration(X, X, X, X, 0.0, 1.0, np.arange(2))

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

      coordinate_ascent_iteration(
        X, Delta, w, lmbd, gamma, p, r.permutation(np.arange(n_pairs)),
      )
