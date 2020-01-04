import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genome_windows_generator", # Replace with your own username
    version="0.0.1",
    author="Tommaso Fontana",
    author_email="tommaso.fontana.96@gmail.com",
    description="Generator of nucleotides windwos from a given assembly",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zommiommy/genome_windows_generator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL2 License",
        "Operating System :: OS Independent",
    ],
    #python_requires='>=3.6',
)