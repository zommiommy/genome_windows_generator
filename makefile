

install:
	~/anaconda3/bin/pip install --user --upgrade .

test:
	~/anaconda3/bin/pytest -s --cov genome_windows_generator --cov-report html

build:
	~/anaconda3/bin/python setup.py sdist

publish:
	echo "Uploading ./dist/$$(ls ./dist | grep .tar.gz | sort | tail -n 1)"
	twine upload "./dist/$$(ls ./dist | grep .tar.gz | sort | tail -n 1)"
