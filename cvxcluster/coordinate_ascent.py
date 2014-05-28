from select import select
import sys

import numpy as np

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

      for l in r.permutation(range(n_pairs)):
        i, j, w_l  = w[l,:]
        Delta[i]  -= lmbd[l]
        Delta[j]  += lmbd[l]
        l_mid      = -0.5 * (Delta[i] - Delta[j] + X[i] - X[j])
        l_new      = problem.project(l_mid[None,:], np.asarray([gamma * w_l]))[0]
        lmbd[l]    = l_new
        Delta[i]  += l_new
        Delta[j]  -= l_new

