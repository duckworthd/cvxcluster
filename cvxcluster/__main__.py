from datetime import datetime
from select import select
from time import time
import json
import logging.config
import os

import numpy as np
import pandas as pd
import pylab as pl

from cvxcluster import *
from cvxcluster.profile import profile


def main(config):

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

  setup   = np.load(config.problem)
  problem = P(setup['X'], setup['gamma'], setup['w'])

  conditions = [UserInput()]
  if config.rtol is not None    : conditions.append(RelativeTolerance(config.rtol))
  if config.atol is not None    : conditions.append(AbsoluteTolerance(config.atol))
  if config.maxiter is not None : conditions.append(MaxIterations(config.maxiter))
  if config.maxtime is not None : conditions.append(MaxTime(config.maxtime))
  if config.mingrad is not None : conditions.append(MinGradient(config.mingrad))

  solution, report = solve(problem, solver, conditions)

  if config.output is not None:
    savestate(config, config.output, solution, report)


def savestate(config, folder, solution, report):
  print "Saving solution to " + folder
  try            : os.makedirs(folder)
  except OSError : pass

  # save problem and solution, for posterity
  np.savez_compressed(
    os.path.join(folder, "problem.npz"),
    X     = solution.problem.X,
    gamma = solution.problem.gamma,
    w     = solution.problem.w,
  )
  np.savez_compressed(
    os.path.join(folder, "solution.npz"),
    lmbd = solution.lmbd
  )

  # save runtime records to CSV
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
  view(solution)
  pl.savefig(os.path.join(folder, "plot.png"))

  # save profiling
  with open(os.path.join(folder, "profile.txt"), 'wb') as f:
    profile.print_stats(f)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser("Convex Clustering Solver")

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

  main(parser.parse_args())
