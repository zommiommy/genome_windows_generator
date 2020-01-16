
PYTHON_PATH=~/anaconda3/bin/

install:
	$(PYTHON_PATH)pip install --user --upgrade .

test:
	$(PYTHON_PATH)pytest -s --cov genome_windows_generator --cov-report html

build:
	$(PYTHON_PATH)python setup.py sdist

publish:
	twine upload "./dist/$$(ls ./dist | grep .tar.gz | sort | tail -n 1)"
