from line_profiler import LineProfiler


__all__ = [
  'profile',
]

class DummyProfiler(object):
  """A fake profiler"""

  def __call__(self, f):
    return f

  def print_stats(self, stream=None):
    pass


profile = LineProfiler()
# profile = DummyProfiler()
