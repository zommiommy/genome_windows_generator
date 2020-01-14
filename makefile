

install:
	~/anaconda3/bin/pip install --user --upgrade .

test:
	~/anaconda3/bin/pytest -s --cov genome_windows_generator --cov-report html

publish:
	~/anaconda3/bin/python setup.py sdist
	~/anaconda3/bin/twine upload $PATH
