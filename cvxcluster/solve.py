import logging
from time import time


__all__ = [
  'solve'
]


LEGEND = (
  "{:^4s} | {:^11s} | {:^11s} | {:^11s} | {:^5s} "
).format("iter", "primal", "dual", "gap", "time")

STATUS = (
  "{i:4d} | {primal: 4.8f} | {dual: 4.8f} | {gap: 4.8f} | {t: 4.2f}"
)


def solve(problem, solver, conditions, lmbd0=None):
  """
  Iteratively solve a problem until an exit condition is met.

  Parameters
  ----------
  problem: Problem
  solver: Solver
  conditions: [Condition]

  Returns
  -------
  solution : Solution
      final solution output by solver
  report : [dict]
      list of a dicts containing checkpoint information about each iteration.
  """
  log = logging.getLogger(__name__)
  if len(conditions) == 0:
    raise ValueError("You have no stopping conditions! This could would run foever...")

  for condition in conditions: condition.reset()
  t0     = time()
  report = []
  for i, solution in enumerate(solver.minimize(problem, lmbd=lmbd0)):

    primal, dual, gap = solution.duality_gap()

    if i == 0: log.info( LEGEND )
    log.info(
      STATUS.format(
        i      = i,
        primal = primal,
        dual   = dual,
        gap    = gap,
        t      = time() - t0,
      )
    )

    report.append({
      "i"      : i,
      "primal" : primal,
      "dual"   : dual,
      "gap"    : gap,
      "t"      : time() - t0,
    })

    if any(c(solution) for c in conditions):
      return solution, report
