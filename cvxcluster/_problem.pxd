import  numpy as np
cimport numpy as np


cdef np.ndarray project(
  np.ndarray[np.float_t, ndim=1] lmbd,
  np.float_t                     eps,
  np.float_t                     p
)

