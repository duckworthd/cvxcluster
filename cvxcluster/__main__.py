from datetime import datetime
from select import select
from time import time
import json
import logging
import logging.config
import os

import numpy as np
import pandas as pd
import pylab as pl

from cvxcluster import *
from cvxcluster.profile import profile


def argparser():
  """Build argument parser for `main`"""

  import argparse

  parser = argparse.ArgumentParser(description="Convex Clustering Solver")

  parser.add_argument("--regularizer", "-r",
    choices = ["inf", "l1", "l2"],
    default = "l2",
    help    = "Regularizer to use between pairs of variables"
  )
  parser.add_argument("--solver", "-s",
    choices = ["coord", "accel-ama", "ama"],
    default = "coord",
    help    = "Solver to use."
  )
  parser.add_argument("--nu", "-n",
    type    = float,
    default = None,
    help    = "Step size to use for AMA-type solvers. If None, nu selected according to the problem instance."
  )
  parser.add_argument("--maxiter",
    type    = float,
    default = None,
    help    = "Maximum number of iterations to run"
  )
  parser.add_argument("--maxtime",
    type    = float,
    default = None,
    help    = "Maximum number of seconds to run until"
  )
  # parser.add_argument("--mingrad",
  #   type    = float,
  #   default = None,
  #   help    = "Minimum value for a subgradient"
  # )
  parser.add_argument("--atol",
    type    = float,
    default = None,
    help    = "Minimum duality gap to run until"
  )
  parser.add_argument("--rtol",
    type    = float,
    default = None,
    help    = "Minimum relative duality gap to run until"
  )
  parser.add_argument("--logconf",
    default = None,
    help    = "Log configuration file (as JSON)"
  )
  parser.add_argument("--output", "-o",
    default = None,
    help    = "Folder to save solution details to."
  )
  parser.add_argument("problem",
    help = "Path to npz file containing problem setup -- specifically, an X, gamma, and w"
  )

  return parser


def setup(config):
  """Setup problem, solver, and conditions for a single run"""
  if config.logconf is not None:
    with open(config.logconf, 'rb') as f:
      logging.config.dictConfig(json.load(f))

  if config.regularizer == 'inf':
    class P(Problem, L_Inf): pass
  if config.regularizer == 'l2':
    class P(Problem, L2): pass
  if config.regularizer == 'l1':
    class P(Problem, L1): pass

  if config.solver == 'ama'       : solver = AMA(config.nu)
  if config.solver == 'accel-ama' : solver = AcceleratedAMA(config.nu)
  if config.solver == 'coord'     : solver = CoordinateAscent()

  setup   = load(config.problem)
  problem = P(setup['X'], setup['gamma'], setup['w'])

  conditions = [UserInput()]
  if config.rtol is not None    : conditions.append(RelativeTolerance(config.rtol))
  if config.atol is not None    : conditions.append(AbsoluteTolerance(config.atol))
  if config.maxiter is not None : conditions.append(MaxIterations(config.maxiter))
  if config.maxtime is not None : conditions.append(MaxTime(config.maxtime))
  # if config.mingrad is not None : conditions.append(MinGradient(config.mingrad))

  return (problem, solver, conditions)


def run(problem, solver, conditions):
  """Solve a problem instance"""
  solution, report = solve(problem, solver, conditions)
  return (solution, report)


def savestate(config, solution, report,
    problem_=True, solution_=True, report_=True, plot_=True, profiling_=True,
  ):
  """
  Save various bits of information about a cvxcluster run to disk.

  Parameters
  ----------
  config : argparse.Namespace
      CLI configuration
  solution : Solution
      Solution to run
  report : [dict]
      runtime and convergence information from the run
  problem_ : bool, optional
      save problem variables to disk?
  solution_ : bool, optional
      save solution variables to disk?
  report_ : bool, optional
      save  report to CSV file?
  plot_ : bool, optional
      save plot of 2D projection of original points and final outputs?
  profiling_ : bool, optional
      save line-profiler output?
  """
  folder = config.output
  if not folder: return

  log    = logging.getLogger(__name__)
  log.info("Saving solution to " + folder)

  # make directory for output
  try            : os.makedirs(folder)
  except OSError : pass

  # save problem and solution, for posterity
  if problem_:
    np.savez_compressed(
      os.path.join(folder, "problem.npz"),
      X     = solution.problem.X,
      gamma = solution.problem.gamma,
      w     = solution.problem.w,
    )
  if solution_:
    np.savez_compressed(
      os.path.join(folder, "solution.npz"),
      lmbd = solution.lmbd
    )

  # save runtime records to CSV
  if report_:
    df = pd.DataFrame.from_records(
      report,
      columns = ['i', 'primal', 'dual', 'gap', 't']
    )
    df['reg']  = config.regularizer
    df['sol']  = config.solver
    df['prob'] = config.problem
    df['date'] = datetime.utcnow().isoformat()
    df.to_csv(
      os.path.join(folder, "report.csv"),
      encoding = "utf-8",
      index    = False,
    )

  # save plot to PNG
  if plot_:
    view(solution)
    pl.savefig(os.path.join(folder, "plot.png"))

  # save profiling
  if profiling_:
    with open(os.path.join(folder, "profile.txt"), 'wb') as f:
      profile.print_stats(f)



if __name__ == '__main__':

  config                      = argparser().parse_args()
  problem, solver, conditions = setup(config)
  solution, report            = run(problem, solver, conditions)

  savestate(config, solution, report)
