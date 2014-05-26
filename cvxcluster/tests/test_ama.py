import unittest

from cvxcluster import AMA, AcceleratedAMA
from . import SolverTests


class AMATests(SolverTests, unittest.TestCase):

  def setUp(self):
    self.solver = AMA(nu=0.05)


class AcceleratedAMATests(SolverTests, unittest.TestCase):

  def setUp(self):
    self.solver = AcceleratedAMA(nu=0.05)
