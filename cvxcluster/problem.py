from collections import defaultdict
import time

import numpy as np

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
      self._v = np.zeros(self.lmbd.shape)
      for l, (i, j, w_l) in enumerate(iterrows(self.problem.w)):
        self._v[l] = self.u[i] - self.u[j]
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
    self.X     = X
    self.gamma = gamma
    self.w     = w

    self.n_samples  = X.shape[0]
    self.n_features = X.shape[1]
    self.n_pairs    = w.shape[0]

  def objective(self, u):
    X, gamma, w = self.X, self.gamma, self.w
    n_samples   = self.n_samples
    n_features  = self.n_features
    n_pairs     = self.n_pairs

    result = 0.0
    for i in range(n_samples):
      result += 0.5 * (np.linalg.norm(X[i] - u[i], 2) ** 2)

    for l, (i, j, w_l) in enumerate(iterrows(w)):
      result += gamma * w_l * self.norm(u[i] - u[j])

    return result

  def objective_subgradient(self, u):
    X, gamma, w = self.X, self.gamma, self.w
    n_samples   = self.n_samples

    result = u - X

    for l, (i, j, w_l) in enumerate(iterrows(w)):
      g = gamma * w_l * self.norm_subgradient(u[i] - u[j])
      result[i] += g
      result[j] -= g

    return result

  def dual(self, Delta, lmbd):
    X, gamma, w = self.X, self.gamma, self.w
    n_samples   = self.n_samples

    result = 0.0

    for i in range(n_samples):
      result -= 0.5 * (np.linalg.norm(Delta[i]) ** 2)

    for l, (i, j, w_l) in enumerate(iterrows(w)):
      result -= lmbd[l].dot(X[i] - X[j])

    # XXX technically, I should also iterate over all l to make sure lmbd[l] is
    # within the dual-norm ball imposed by w_l * gamma, but numerical errors
    # abound.

    return result

  def u(self, Delta):
    return self.X + Delta

  def Delta(self, lmbd):
    n_samples, n_features = self.n_samples, self.n_features
    Delta                 = np.zeros( (n_samples, n_features) )
    for l, (i, j, w_l) in enumerate(iterrows(self.w)):
      Delta[i] += lmbd[l]
      Delta[j] -= lmbd[l]
    return Delta


class L2(object):
  """L2 norm mixin for Problem"""

  def norm(self, v):
    """|| v ||_2"""
    return np.linalg.norm(v, 2)

  def norm_subgradient(self, v):
    n = self.norm(v)
    if n == 0 : return np.zeros(v.shape)
    else      : return v / n

  def dual_norm(self, l):
    """|| l ||_2"""
    return self.norm(l)

  def project(self, l, e):
    """Project onto {v : ||v||_2 <= e}"""
    if self.dual_norm(l) <= e:
      return l
    else:
      # subtract off a tiny bit to keep the dual norm <= e
      if e > 0:
        e = max(0, e - 1e-15)
      return e * (l / self.dual_norm(l))


class L_Inf(object):
  """L-infinity norm mixin for Problem"""

  def norm(self, v):
    return np.linalg.norm(v, np.inf)

  def norm_subgradient(self, v):
    n = self.norm(v)
    if n == 0 : return np.zeros(v.shape)
    else      : return np.sign(v) * (np.abs(v) >= n)

  def dual_norm(self, l):
    return np.linalg.norm(l, 1)

  def project(self, l, e):
    """Project onto {v : ||v||_1 <= e}

      min_{y} 0.5 ||y - l||_2^2
      s.t.    ||y||_{*} <= e

    Reference:
      John Duchi et al, "Efficient Projections onto the l_1-Ball for Learning
        in High Dimensions"
    """
    e = max(0, e - 1e-15)   # cushion for numerical errors

    ll = np.sort(np.abs(l))[::-1]
    cs = np.cumsum(ll)
    vtheta = (cs-e)/(np.arange(len(ll))+1.0)
    i = np.argmin(ll-vtheta >= 0)-1
    prevtheta = max(0, vtheta[i])
    a = np.maximum(0, np.abs(l) - prevtheta) * np.sign(l)
    return a


class L1(object):
  """L1 norm mixin for Problem"""

  def norm(self, v):
    return np.linalg.norm(v, 1)

  def norm_subgradient(self, v):
    return np.sign(v)

  def dual_norm(self, l):
    return np.linalg.norm(l, np.inf)

  def project(self, l, e):
    """Project onto {v : ||v||_inf <= e}"""
    e = max(0, e - 1e-15)
    l = np.array(l)
    l[l < -e] = -e
    l[l >  e] = e
    return l
