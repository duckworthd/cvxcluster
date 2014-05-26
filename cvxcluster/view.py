
import numpy as np
import pylab as pl


__all__ = [
  'view',
]

def view(solution):
  """
  Plot a 2D projection of the original points and the discovered solution. Each
  original point is drawn as a square.  Each "center" associated with a point
  is drawn as a circle and has a line drawn between it and its corresponding
  point. Points/Centers/Lines with the same color are in the same cluster.
  """
  X        = solution.problem.X
  C        = solution.u
  labels   = solution.clusters
  maxlabel = np.max(labels).astype(float) + 1

  if X.shape[1] > 2:
    X, U = pca(X, n=2)
    C = C.dot(U)
  if X.shape[1] < 2:
    X = np.hstack([X, np.zeros(len(X))])
    U = np.hstack([U, np.zeros(len(X))])

  cmap = pl.get_cmap("Paired")

  # draw original points
  pl.scatter(X[:,0], X[:,1], c=labels/maxlabel, cmap=cmap, marker='s')

  # draw centers
  pl.scatter(C[:,0], C[:,1], c=labels/maxlabel, cmap=cmap, marker='o')

  # draw line between points, centers
  for i in range(len(X)):
    x = X[i]
    c = C[i]
    l = labels[i]
    s = np.vstack([x, c])
    pl.plot(s[:,0], s[:,1], color=cmap(l / maxlabel))


def pca(X, n=2):
  "Project X onto its n most varying dimensions"
  S = np.cov(X.T)

  # UDU' = S
  _, U = np.linalg.eig(S)

  # first n eigenvectors
  U = U[:,0:n]

  # project onto top n eigenvectors
  return X.dot(U), U
