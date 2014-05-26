"""
Conditions under which to stop iterating on solutions.
"""
from time import time
from select import select
import logging
import sys

import numpy as np

__all__ = [
  'RelativeTolerance',
  'AbsoluteTolerance',
  'MaxIterations',
  'MaxTime',
  'MinGradient',
  'UserInput',
]


class RelativeTolerance(object):

  def __init__(self, rtol):
    self.rtol    = rtol

    # initial gap
    self.primal0 = None
    self.dual0   = None

    # best gap so far
    self.primal  = None
    self.dual    = None

  def __call__(self, solution):
    primal, dual, _ = solution.duality_gap()

    if self.primal0 is None: self.primal0 = primal
    if self.dual0   is None: self.dual0   = dual

    if self.primal is None: self.primal = primal
    if self.dual   is None: self.dual   = dual

    self.primal = min(self.primal, primal)
    self.dual   = max(self.dual  , dual  )

    gap0 = self.primal0 - self.dual0
    gap  = self.primal  - self.dual

    if gap0 == 0: return True

    rgap = 1 - (gap0 - gap)/gap0
    return rgap < self.rtol

  def reset(self):
    self.primal0 = self.dual0 = None
    self.primal  = self.dual  = None


class AbsoluteTolerance(object):

  def __init__(self, atol):
    self.atol    = atol

    # best gap so far
    self.primal  = None
    self.dual    = None

  def __call__(self, solution):
    primal, dual, _ = solution.duality_gap()

    if self.primal is None: self.primal = primal
    if self.dual   is None: self.dual   = dual

    self.primal = min(self.primal, primal)
    self.dual   = max(self.dual  , dual  )

    gap  = self.primal - self.dual

    return gap < self.atol

  def reset(self):
    self.primal = self.dual = None


class MaxIterations(object):
  def __init__(self, maxiter):
    self.maxiter   = maxiter
    self.iteration = 0

  def __call__(self, solution):
    self.iteration += 1
    return self.iteration >= self.maxiter

  def reset(self):
    self.iteration = 0


class MaxTime(object):
  def __init__(self, maxtime):
    self.maxtime = maxtime
    self.start   = None

  def __call__(self, solution):
    if self.start is None:
      self.start = time()
    now = time()
    return now - self.start > self.maxtime

  def reset(self):
    self.start = None


class MinGradient(object):

  def __init__(self, norm):
    self.norm = norm

  def __call__(self, solution):
    gradient = solution.problem.objective_subgradient(solution.u)
    size     = np.product(gradient.shape)

    norm     = np.linalg.norm(gradient.reshape((size,))) ** 2.0

    logging.getLogger(__name__).info("gradient norm: {}".format(norm))

    return norm <= self.norm

  def reset(self):
    pass


class UserInput(object):

  def __call__(self, solution):
    log = logging.getLogger(__name__)
    if select([sys.stdin], [], [], 0)[0]:
      line = sys.stdin.readline().strip().lower()
      if line == 'quit':
        log.info("Exiting on user input")
        return True
      else:
        return False
    else:
      return False

  def reset(self):
    pass
