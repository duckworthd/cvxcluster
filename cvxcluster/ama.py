from collections import defaultdict
from select import select
import logging
import sys

import numpy as np

from .problem import Solution
from .profile import profile
from .utils import *


__all__ = [
  'AMA',
  'AcceleratedAMA',
]


class AMA(object):
  """Solver for Convex Clustering using AMA

  Employs the Alternating Minimization Algorithm (AMA) to solve the Convex
  Clustering problem, as detailed in Chi et al. 2013. This is equivalent to
  Proximal Gradient Ascent on the dual.

  Parameters
  ----------
  nu : float
      step size
  """

  def __init__(self, nu):
    self.nu = nu

  @profile
  def minimize(self, problem, lmbd=None):
    X, gamma, w           = problem.X, problem.gamma, problem.w
    nu                    = self.nu
    n_samples, n_features = X.shape
    n_pairs               = len(w)

    if lmbd is None:
      lmbd = np.zeros( (n_pairs, n_features) )
    if nu is None:
      nu = fit_nu(problem)

    while True:
      yield Solution(problem, lmbd=lmbd)

      Delta = problem.Delta(lmbd)

      # compute gradient
      i, j = w[:,0].astype(int), w[:,1].astype(int)
      g = X[i] + Delta[i] - X[j] - Delta[j]
      # g = np.zeros( (n_pairs, n_features) )
      # for l, (i, j, w_l) in enumerate(iterrows(w)):
      #   g[l] = X[i] + Delta[i] - X[j] - Delta[j]

      # take a (projected) gradient step
      lmbd_ = np.zeros(lmbd.shape)
      lmbd_ = problem.project(lmbd - nu * g, gamma * w[:,2])
      # for l, (i, j, w_l) in enumerate(iterrows(w)):
      #   lmbd_[l] = problem.project(lmbd[l] - nu * g[l], gamma * w_l)

      lmbd = lmbd_


class AcceleratedAMA(object):
  """Solver for Convex Clustering using AMA

  Realizing that AMA is the same as Proximal Gradient Ascent on the Dual
  problem, this solver uses _Accelerated_ Proximal Gradient Ascent to get
  a better asymptotic convergence rate.

  Parameters
  ----------
  nu : float
      step size
  """

  def __init__(self, nu):
    self.nu = nu

  @profile
  def minimize(self, problem, lmbd=None):
    X, gamma, w           = problem.X, problem.gamma, problem.w
    nu                    = self.nu
    n_samples, n_features = problem.n_samples, problem.n_features
    n_pairs               = problem.n_pairs

    if lmbd is None:
      lmbd = np.zeros( (n_pairs, n_features) )
    if nu is None:
      nu = fit_nu(problem)

    alpha  = 1.0            # \alpha^{m-1}
    lmbd__ = lmbd_ = lmbd   # \tilde{\lambda}^{m-1} and \tilde{\lambda}^{m}

    while True:

      yield Solution(problem, lmbd=lmbd_)

      Delta = problem.Delta(lmbd)

      # compute gradient
      i, j = w[:,0].astype(int), w[:,1].astype(int)
      g = X[i] + Delta[i] - X[j] - Delta[j]
      # g = np.zeros( (n_pairs, n_features) )
      # for l, (i, j, w_l) in enumerate(iterrows(w)):
      #   g[l] = X[i] + Delta[i] - X[j] - Delta[j]

      # take a (projected) gradient step
      lmbd_ = np.zeros(lmbd.shape)
      lmbd_ = problem.project(lmbd - nu * g, gamma * w[:,2])
      # for l, (i, j, w_l) in enumerate(iterrows(w)):
      #   lmbd_[l] = problem.project(lmbd[l] - nu * g[l], gamma * w_l)

      # construct the actual iterate using a combination of the previous step
      # and the last iteration.
      alpha, alpha_ = (1.0 + np.sqrt(1 + 4 * alpha * alpha)) / 2.0, alpha

      lmbd   = lmbd_ + (alpha_ / alpha) * (lmbd_ - lmbd__)
      lmbd__ = lmbd_


@profile
def fit_nu(problem):
  """Fit learning rate to problem instance."""
  log = logging.getLogger(__name__)
  w = problem.w
  degrees = defaultdict(lambda: 0)
  for i, j, _ in iterrows(w):
    degrees[i] += 1
    degrees[j] += 1

  if len(degrees) == 0:
    return 0.0
  else:
    # Chi 2013 proves that so long as nu < 2 / \rho(A^T A), AMA is guaranteed
    # to converge (\rho(...) is the largest eigenvalue of a matrix, and A is
    # the matrix used to define the equality constraint Ax + by = c in the
    # original problem formulation).
    #
    # He further shows the relationship between \rho(A^T A) = \rho(L), the
    # Laplacian of the graph defined by A. \rho(L) is upper bounded by
    # `candidate` below.
    candidate = max(degrees[i] + degrees[j] for i, j, _ in iterrows(w))
    nu        = min(candidate, problem.n_samples)
    log.info("Using nu = 1.0 / {} ({} samples)".format(nu, problem.n_samples))
    return 1.0 / nu
