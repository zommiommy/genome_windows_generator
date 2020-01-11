

install:
	~/anaconda3/bin/pip install --user --upgrade .

test:
	~/anaconda3/bin/pytest -s --cov windows_generator --cov-report html
