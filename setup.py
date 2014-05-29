from setuptools import setup, find_packages

def version(name):
  import os
  folder   = os.path.split(__file__)[0]
  mirai    = os.path.join(folder, name)
  _version = os.path.join(mirai, "_version.py")
  env      = {}
  execfile(_version, env)
  return env['__version__']


if __name__ == '__main__':
  setup(
      name              = 'cvxcluster',
      version           = version('cvxcluster'),
      author            = 'Daniel Duckworth',
      author_email      = 'duckworthd@gmail.com',
      description       = "Convex Clustering solvers",
      license           = 'BSD',
      keywords          = 'convex optimization numerical research',
      url               = 'http://github.com/duckworthd/cvxcluster',
      packages          = find_packages(),
      classifiers       = [
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
      ],
      setup_requires    = [
        "setuptools>=3.4.4",
      ],
      install_requires  = [
        "numpy>=1.7.0",
        "scipy>=0.13.0",
      ],
      tests_require     = [
        "nose>=1.3.1",
      ],
      test_suite = "nose.collector",

      # non-python files to include
      package_data         = {
        'cvxcluster.tests' : ['simple/*.npz'],
        'cvxcluster'       : ['*.pyx'],
      },
      include_package_data = True,
  )
