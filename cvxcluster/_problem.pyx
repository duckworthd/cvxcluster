import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray project(
    np.ndarray[np.float_t, ndim=1] lmbd,
    np.float_t                     eps,
    np.float_t                     p
  ):
  if   p == 1.0:
    return project_1  (lmbd, eps)
  elif p == 2.0:
    return project_2  (lmbd, eps)
  elif p == np.inf:
    return project_inf(lmbd, eps)

  # this should never happen
  return None


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray project_2(
    np.ndarray[np.float_t, ndim=1] lmbd,
    double                         eps,
  ):
  # Perform a projection into the L_2 ball of radius eps

  cdef double norm = np.linalg.norm(lmbd)

  if norm < eps : return lmbd
  else          : return (eps/norm) * lmbd


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray project_1(
    np.ndarray[np.float_t, ndim=1] lmbd,
    np.float_t                     eps
  ):
  # project onto the L_{inf} ball of radius eps

  cdef int n_pairs = len(lmbd)
  cdef int i
  cdef np.float_t v

  for i in range(n_pairs):
    v = lmbd[i]
    if   v < -eps:
      lmbd[i] = -eps
    elif v >  eps:
      lmbd[i] =  eps

  return lmbd


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray project_inf(
    np.ndarray[np.float_t, ndim=1] lmbd,
    np.float_t                     eps
  ):
  # project into the L_1 ball of radius eps. This involves a lot of calls to
  # numpy, which has Python-function-call overhead. In other words, don't
  # expect spectacular performace :/

  cdef np.ndarray[np.float_t, ndim=1] ll, cs, vtheta, a
  cdef int i
  cdef np.float_t prevtheta

  ll        = np.sort(np.abs(lmbd))[::-1]
  cs        = np.cumsum(ll)
  vtheta    = (cs-eps)/(np.arange(len(ll))+1.0)
  i         = np.argmin(ll-vtheta >= 0)-1
  if i < 0: i = len(lmbd) - 1
  prevtheta = max(0, vtheta[i])
  a         = np.maximum(0, np.abs(lmbd) - prevtheta) * np.sign(lmbd)
  return a
