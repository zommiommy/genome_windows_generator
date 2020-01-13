

install:
	~/anaconda3/bin/pip install --user --upgrade .

test:
	~/anaconda3/bin/pytest -s --cov genome_windows_generator --cov-report html
