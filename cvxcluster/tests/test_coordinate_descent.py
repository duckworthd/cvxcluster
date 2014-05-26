import unittest

from cvxcluster import CoordinateAscent
from . import SolverTests


class CoordinateAscentTests(SolverTests, unittest.TestCase):

  def setUp(self):
    self.solver = CoordinateAscent()
