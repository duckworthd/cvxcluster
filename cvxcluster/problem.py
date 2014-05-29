from collections import defaultdict
import time

import numpy as np

from .profile import profile
from .utils import *


__all__ = [
  'Solution',
  'Problem',
  'L2',
  'L_Inf',
  'L1',
]

class Solution(object):
  """
  A solution to the Convex Clustering problem.

  Parameters
  ----------
  problem : Problem
      problem being solved
  lmbd : [n_pairs, n_features] array
      dual variables
  u : None or [n_samples, n_features] array, optional
      primal variables. if None, will be constructed from dual variables.
  Delta : None or [n_samples, n_features] array, optional
      used to compute dual objective. if None, will be constructed from dual variables.
  """

  def __init__(self, problem, lmbd, u=None, v=None, Delta=None):
    self.problem = problem
    self.lmbd    = lmbd
    self.now     = time.time()
    self._u      = u
    self._v      = v
    self._Delta  = Delta

  @property
  def u(self):
    if self._u is None:
      self._u = self.problem.u(self.Delta)
    return self._u

  @property
  def v(self):
    if self._v is None:
      i, j = self.problem.w[:,0].astype(int), self.problem.w[:,1].astype(int)
      self._v = self.u[i] - self.u[j]
    return self._v

  @property
  def clusters(self):
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    n_samples  = self.problem.n_samples
    n_features = self.problem.n_features

    # two centers are "the same" if they're much closer to each other than any
    # 2 data points are (excepting overlapping points)
    norm      = self.problem.norm
    X, w      = self.problem.X, self.problem.w
    distances = [norm(X[i] - X[j]) for (i, j, _) in iterrows(w)]
    distances = [d for d in distances if d > 0]
    if len(distances) == 0:
      epsilon = 1e-5  # arbitrary
    else:
      epsilon = min(d for d in distances if d > 0) * 1e-2

    edgelist = []
    for l,(i,j,_) in enumerate(iterrows(self.problem.w)):
      if np.linalg.norm(self.v[l]) / n_features < epsilon:
        edgelist.append( (1, i, j) )
        edgelist.append( (1, j, i) )
    if len(edgelist) > 0:
      vals, rows, cols = zip(*edgelist)
    else:
      vals, rows, cols = [], [], []
    adjacency            = csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    n_components, labels = connected_components(adjacency, directed=False)

    return labels

  @property
  def Delta(self):
    if self._Delta is None:
      self._Delta = self.problem.Delta(self.lmbd)
    return self._Delta

  def duality_gap(self):
    """Calculate absolute duality gap"""
    problem  = self.problem
    lmbd     = self.lmbd

    primal   = problem.objective(self.u)
    dual     = problem.dual(self.Delta, lmbd)
    absolute = (primal - dual)

    return primal, dual, primal - dual


class Problem(object):
  '''Convex Clustering problem

  min_{u_i} (1/2) \sum_{i=1}^{p} ||x_i - u_i||_2^2
                  + \gamma \sum_{l} w_{l} ||u_{l_1} - u_{l_2}||_2

  where the first sum is over all points x_i and the latter is over all pairs
  of points l = l_1, l_2 s.t. l_1 < l_2.

  Requires the implementation of self.project(l, e) which is the projection of
  l onto the set {v : ||v||_{*} <= e} and self.norm(v) = ||v||.

  Parameters
  ----------
  X : [n_samples, n_features] array
      points to cluster, one per row
  gamma : float
      regularization parameter
  w : [ n_pairs, 3 ]
      Triplets of the form (i, j, w) where i and j are 2 rows X and w is the
      strength with which you believe i and j should be clustered together.
  '''

  def __init__(self, X, gamma, w):
    self.X     = X.astype(float)
    self.gamma = gamma
    self.w     = w.astype(float)

    self.n_samples  = X.shape[0]
    self.n_features = X.shape[1]
    self.n_pairs    = w.shape[0]

  @profile
  def objective(self, u):
    X, gamma, w = self.X, self.gamma, self.w
    n_samples   = self.n_samples
    n_features  = self.n_features
    n_pairs     = self.n_pairs

    w        = self.w[:,0:2].astype(int)
    w_l      = self.w[:,2]
    i, j     = w[:,0], w[:,1]
    result   = 0.5 * np.sum(np.linalg.norm(X - u, 2, axis=1) ** 2.0)
    result  += gamma * np.sum(self.norm(u[i] - u[j]) * w_l)

    # # the above is just a vectorized version of this
    # result = 0.0
    # for i in range(n_samples):
    #   result += 0.5 * (np.linalg.norm(X[i] - u[i], 2) ** 2)

    # for l, (i, j, w_l) in enumerate(iterrows(w)):
    #   result += gamma * w_l * self.norm(u[i] - u[j])

    return result

  @profile
  def dual(self, Delta, lmbd):
    X, gamma, w = self.X, self.gamma, self.w
    n_samples   = self.n_samples

    i       = w[:,0].astype(int)
    j       = w[:,1].astype(int)
    result  = -0.5 * np.sum(np.linalg.norm(Delta, axis=1) ** 2.0)
    result += -np.einsum('ij,ij->', lmbd, X[i]-X[j])

    # XXX technically, I should also iterate over all l to make sure lmbd[l] is
    # within the dual-norm ball imposed by w_l * gamma, but numerical errors
    # abound.

    # # the above is just a vectorized version of this.
    # result = 0.0
    # for i in range(n_samples):
    #   result -= 0.5 * (np.linalg.norm(Delta[i]) ** 2)

    # for l, (i, j, w_l) in enumerate(iterrows(w)):
    #   result -= lmbd[l].dot(X[i] - X[j])

    return result

  def u(self, Delta):
    return self.X + Delta

  @profile
  def Delta(self, lmbd):
    n_samples  = self.n_samples
    n_features = self.n_features

    Delta      = np.zeros( (n_samples, n_features) )
    w          = self.w.astype(int)[:,0:2]
    i, j       = w[:,0], w[:,1]

    np.add     .at(Delta, i, lmbd)
    np.subtract.at(Delta, j, lmbd)

    # # the cryptic business above here is just an efficient way of doing this,
    # for l, (i, j, w_l) in enumerate(iterrows(self.w)):
    #   Delta[i] += lmbd[l]
    #   Delta[j] -= lmbd[l]

    return Delta


class L2(object):
  """L2 norm mixin for Problem"""

  _norm = 2.0

  def norm(self, v):
    """|| v ||_2"""
    v = np.atleast_2d(v)
    return np.linalg.norm(v, self._norm, axis=1)

  def dual_norm(self, l):
    """|| l ||_2"""
    return self.norm(l)

  @profile
  def project(self, lmbd, eps):
    """Project onto {v : ||v||_2 <= e}"""
    lmbd = np.atleast_2d(lmbd)
    eps  = np.atleast_1d(eps)

    norms      = np.linalg.norm(lmbd, axis=1)
    indicators = (norms <= eps)
    projected = (
      (1-indicators[:,None]) * (eps[:,None] * lmbd / norms[:,None]) +
      (  indicators[:,None]) * lmbd
    )

    return projected


class L_Inf(object):
  """L-infinity norm mixin for Problem"""

  _norm = np.inf

  def norm(self, v):
    v = np.atleast_2d(v)
    return np.linalg.norm(v, np.inf, axis=1)

  def dual_norm(self, l):
    l = np.atleast_2d(l)
    return np.linalg.norm(l, 1, axis=1)

  @profile
  def project(self, lmbd, eps):
    """Project onto {v : ||v||_1 <= e}

      min_{y} 0.5 ||y - l||_2^2
      s.t.    ||y||_{*} <= e

    Reference:
      John Duchi et al, "Efficient Projections onto the l_1-Ball for Learning
        in High Dimensions"
    """
    lmbd = np.atleast_2d(lmbd)
    eps  = np.atleast_1d(eps)

    ll        = np.sort(np.abs(lmbd), axis=1)[:, ::-1]
    cs        = np.cumsum(ll, axis=1)
    vtheta    = (cs - eps[:,None]) / (np.arange(ll.shape[1]) + 1.0)[None,:]
    i         = np.argmin(ll - vtheta >= 0, axis=1) - 1
    i         = np.where(i < 0, ll.shape[1]-1, i)
    prevtheta = np.maximum(0.0, np.choose(i, vtheta.T))
    a         = np.maximum(0.0, np.abs(lmbd) - prevtheta[:,None]) * np.sign(lmbd)

    return a


class L1(object):
  """L1 norm mixin for Problem"""

  _norm = 1.0

  def norm(self, v):
    v = np.atleast_2d(v)
    return np.linalg.norm(v, 1, axis=1)

  def dual_norm(self, l):
    l = np.atleast_2d(l)
    return np.linalg.norm(l, np.inf, axis=1)

  @profile
  def project(self, lmbd, eps):
    """Project onto {v : ||v||_inf <= e}"""
    lmbd = np.atleast_2d(lmbd)
    eps  = np.atleast_1d(eps)

    lmbd = lmbd.copy()
    i = (lmbd < -eps[:,None])
    lmbd = lmbd * (1-i) + -eps[:,None] * i

    j = (lmbd >  eps[:,None])
    lmbd = lmbd * (1-j) +  eps[:,None] * j

    return lmbd
