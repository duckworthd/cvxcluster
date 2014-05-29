PROJECT = cvxcluster
ARGS    = 


default: clean test

dependencies:
	test -e /tmp/$(PROJECT) || virtualenv /tmp/$(PROJECT)
	. /tmp/$(PROJECT)/bin/activate; pip install -r requirements.txt

test: dependencies
	. /tmp/$(PROJECT)/bin/activate; python -m nose

run: dependencies
	. /tmp/$(PROJECT)/bin/activate; python -m $(PROJECT) ${ARGS}

clean:
	find $(PROJECT) -iname '*.pyc' | xargs rm
	find $(PROJECT) -iname '*.so'  | xargs rm
	find $(PROJECT) -iname '*.c'   | xargs rm
	rm -rf *.pyc *.pyo *.egg-info dist build target

clean-env: clean
	rm -rf /tmp/$(PROJECT)

register:
	. /tmp/$(PROJECT)/bin/activate; python setup.py register

deploy: test
	. /tmp/$(PROJECT)/bin/activate; python setup.py sdist upload

build:
	. /tmp/$(PROJECT)/bin/activate; python setup.py build
