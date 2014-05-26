import logging; log = logging.getLogger(__name__)

import numpy as np

from .solve import solve
from .conditions import RelativeTolerance


__all__ = [
  'clusterpath',
]

def clusterpath(problem0, solver, conditions, gamma0=1e-3, step=1.2):
  """
  Calculate the "clusterpath" -- the sequence of gamma values that cause the
  number of clusters to change -- by guess-and-check.

  Parameters
  ----------
  problem0 : Problem
      a convex clustering problem
  solver : Solver
      a solver for convex clustering
  gamma0 : float > 0
      initial gamma value
  step : float > 1
      gamma <- gamma * step at each iteration
  rtol : float < 1
      what percentage the duality gap must decrease before calling a problem
      "solved"
  """
  gamma      = gamma0
  n_clusters = problem0.n_samples
  lmbd0      = None

  while n_clusters > 1:
    problem     = problem0.__class__(problem0.X, gamma, problem0.w)
    solution, _ = solve(problem, solver, conditions, lmbd0=lmbd0)
    n_clusters_ = 1 + np.max(solution.clusters)

    log.info( '{:4.8f} | {:4d}'.format(gamma, n_clusters_) )

    if n_clusters > n_clusters_:
      n_clusters  = n_clusters_
      yield gamma, n_clusters_

    gamma *= step
    lmbd0  = solution.lmbd

    for condition in conditions:
      condition.reset()
