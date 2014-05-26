import numpy as np


__all__ = [
  'NormalSampler',
  'nearest_neighbors',
]

class NormalSampler(object):

  def __init__(self, r, k=1, dim=2):
    self.r   = r
    self.k   = k
    self.dim = dim

  def sample_points(self, n_samples):
    X = []
    for i in range(self.k):
      center = 10 * self.r.randn(self.dim)
      points = self.r.randn(n_samples, self.dim) + center
      X.append(points)
    return np.vstack(X)


def nearest_neighbors(X, k, phi):
  """
  Construct pairwise weights by finding the k nearest neighbors to each point
  and assigning a Gaussian-based distance.

  In particular, w_{i,j} = exp(-phi ||x_i - x_j||_2^2) if j is one of i's k
  closest neighbors.

  If j is one of i's closest neighbors and i is one of j's closest members, the
  edge will only appear once with i < j.

  Parameters
  ----------
  X : [n_samples, n_dim] array
  k : int
      number of neighbors for each sample in X
  phi : float
      non-negative integer dictating how much weight to assign between pairs of points
  """
  from scipy.spatial import KDTree

  tree = KDTree(X)

  weights = []
  for i, x in enumerate(X):
    distances, indices = tree.query(x, k+1)
    for (d, j) in zip(distances, indices)[1:]:
      d = np.exp( -phi * d * d )
      if d == 0: continue
      if i < j:
        weights.append( (i, j, d) )
      else:
        weights.append( (j, i, d) )
  weights = sorted(weights, key=lambda r: (r[0], r[1]))
  return unique_rows(np.asarray(weights))


def unique_rows(a):
  """Get unique rows from matrix a"""
  b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
  _, idx = np.unique(b, return_index=True)
  return a[idx]
