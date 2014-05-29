import math
import numpy as np

try:
  from numba import njit, jit
except ImportError:
  njit = jit = lambda s: lambda x: x


@njit("void(f8[:])")
def sillysort(arr):
  # sort an array by swapping. asymptotically shitty. Should be replaced with
  # mergesort at some point, if I can somehow get it to place nicely with
  # numba's njit.
  ndim = arr.shape[0]
  for i in range(ndim):
    for j in range(i,ndim):
      if arr[i] > arr[j]:
        tmp    = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp


@njit("f8(f8)")
def sign(v):
  if v < 0: return -1
  if v > 0: return  1
  else    : return  0


@njit("f8(f8[:])")
def sum_(arr):
  r = 0.0
  for i in range(arr.shape[0]):
    r += arr[i]
  return r


@njit("void(f8[:],f8,f8[:],f8[:])")
def project_1(lmbd, eps, out, tmp):
  ndim = lmbd.shape[0]

  for i in range(ndim):
    if lmbd[i] < -eps:
      out[i] = -eps
    elif lmbd[i] > eps:
      out[i] =  eps
    else:
      out[i] = lmbd[i]


@njit("void(f8[:],f8,f8[:],f8[:])")
def project_2(lmbd, eps, out, tmp):
  norm = 0.0
  ndim = lmbd.shape[0]
  for i in range(ndim):
    norm += lmbd[i] * lmbd[i]

  norm = math.sqrt(norm)

  if norm > eps :
    np.multiply(lmbd, eps/norm, out)
  else:
    for i in range(ndim):
      out[i] = lmbd[i]


@njit("void(f8[:],f8,f8[:],f8[:])")
def project_inf(lmbd, eps, out, tmp):
  ndim = lmbd.shape[0]

  # sort from largest to smallest absolute value
  for i in range(ndim):
    tmp[i] = -abs(lmbd[i])
  sillysort(tmp)
  for i in range(ndim):
    tmp[i] *= -1

  # have we already hit that breakpoint we need? I can't break or return in the
  # middle of a loop or numba's JIT loop magic won't work.
  found = False

  # will contain cumulative sum of all temporary values up to and including
  # index i.
  cum   = 0.0
  for i in range(ndim):
    cum += tmp[i]

    if (tmp[i] - (cum-eps)/(i+1.0) < 0 or i == ndim-1) and not found:
      found = True

      if i == 0 or (i == ndim-1 and tmp[i] - (cum-eps)/(i+1.0) >= 0):
        theta = (sum_(tmp)    - eps) / (ndim+0.0)
      else:
        theta = (cum - tmp[i] - eps) / (i   +0.0)
      theta = max(theta, 0.0)

      for j in range(ndim):
        out[j] = max(0, abs(lmbd[j])-theta) * sign(lmbd[j])


@jit("void(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8,f8)")
def coordinate_ascent_iteration(X, Delta, w, lmbd, gamma, p):
  # numpy arrays can't be constructed in numba's nopython mode. I initialize
  # these temporary arrays here, just to get things going.
  tmp   = np.empty_like(X[0])
  tmp2  = np.empty_like(X[0])

  for l in range(w.shape[0]):
    _iterate(X, Delta, w, lmbd, gamma, p, l, tmp, tmp2)


@njit("void(f8[:,:],f8[:,:],f8[:,:],f8[:,:],i4,f8,f8,f8[:],f8[:])")
def _iterate(X, Delta, w, lmbd, gamma, p, l, tmp, tmp2):
  # sweet baby jesus this is ugly. the things I do for JIT optimization.
  # Sheesh.
  i    = int(w[l,0])
  j    = int(w[l,1])
  w_l  = w[l,2]
  ndim = X.shape[1]

  # remove lmbd[l] from Delta[i] and Delta[j]
  for _ in range(ndim):
    tmp[_] = lmbd[l,_]

  for _ in range(ndim):
    Delta[i,_] -= tmp[_]
    Delta[j,_] += tmp[_]

  # Find the ideal least squares solution
  for _ in range(ndim):
    tmp[_] = -0.5 * (Delta[i,_] + X[i,_] - Delta[j,_] - X[j,_])

  # project back onto the dual-norm ball
  if p == 1.0     : project_1  (tmp, gamma * w_l, tmp, tmp2)
  if p == 2.0     : project_2  (tmp, gamma * w_l, tmp, tmp2)
  if math.isinf(p): project_inf(tmp, gamma * w_l, tmp, tmp2)

  # re-add the new lmbd[l] to Delta[i] and Delta[j]
  for _ in range(ndim):
    lmbd[l,_]   = tmp[_]
    Delta[i,_] += tmp[_]
    Delta[j,_] -= tmp[_]

