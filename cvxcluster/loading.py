from os.path import join, split, abspath

import numpy as np


__all__ = [
  'load',
  'save',
]


def load(folder, fname="problem.npz"):
  """
  Load a numpy environment by loading all files of the same name in a file
  hierarchy. Variables defined deeper in the file hierarchy "overshadow" ones
  defined earlier.

  Parameters
  ----------
  folder : str
      folder to start in
  fname : str, optional
      name of file to load from each
  """
  folder = abspath(folder)
  result = {}

  while folder and folder != split(folder)[0]:
    try:
      with np.load(join(folder, fname)) as f:
        # load all undefined variables into result object
        for k, v in f.items():
          if k not in result:
            result[k] = v

    except IOError:
      pass  # no file named `fname` in this folder!

    # go up one
    folder = split(folder)[0]

  return result


def save(env, folder, fname="problem.npz", saveonly=None):
  """
  Save values from a numpy environment to disk.

  Parameters
  ----------
  env : dict-like
      Environment containing numpy-storable data to save
  folder : str
      Folder to save to
  fname : str
      Filename to save to
  saveonly : [str], optional
      By default, this method will determine which variables in `env` that need
      to be saved in order to reconstruct it. This involves loading from the
      current directory. You can specify which variables to store manually
      here.
  """

  if saveonly is None:
    # generate saveonly by checking what variables are new/have changed.
    saveonly = [
      k for k, v in load(split(folder)[0], fname=fname).items()
      if k not in env or not np.all(env[k] == v)
    ]

  # filter out all variables but the ones that differ from parent directories
  env = {k:v for k,v in env.items() if k in saveonly}

  # store to disk
  np.savez_compressed(join(folder, fname), **env)
