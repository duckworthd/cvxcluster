import math
from time import clock


class Timer(object):
  """A simple timer you can start and stop"""

  def __init__(self):
    self.elapsed = 0.0
    self.start_time = float('nan')
    self.running = False

  def start(self):
    """Start the timer"""
    if not self.running:
      self.start_time = clock()
      self.running = True

  def stop(self):
    """Stop the timer and record the elapsed time"""
    if self.running:
      assert not math.isnan(self.start_time)
      now = clock()
      self.elapsed += (now - self.start_time)
      self.running = False
      self.start_time = float('nan')

  def add(self, time):
    """Add some time to the clock"""
    assert not math.isnan(time)
    self.elapsed += time
