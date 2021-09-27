import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="differential_value_iteration",
    version="0.0.1",
    author="Abhishek Naik",
    author_email="abhisheknaik22296@gmail.com",
    description="Experimental algorithms for differential value iteration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhisheknaik96/differential-value-iteration",
    project_urls={
        "Bug Tracker": "https://github.com/abhisheknaik96/differential-value-iteration/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=["absl-py", "jaxlib", "matplotlib", "numpy", "quantecon"],
)
