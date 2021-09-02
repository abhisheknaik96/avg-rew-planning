# differential-value-iteration
Experiments with new and existing average-reward planning algorithms.

# Prerequisites
- Install JAX https://github.com/google/jax#installation

Possibly just:
```
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```
# Installation
- Clone the github repository to a local directory.

## Developer
- From the root of the repo, `python3 setup.py develop --user

This will install into your user folder.

To check if it is working, you should be able to execute main.py with Python 3.8 or higher.

Eg. `python main.py` or `python3 main.py`

You can uninstall, `python3 setup.py develop --uninstall`

To run the tests, from the root of the repo (after running setup.py):
	`python3 src/differential_value_iteration/import_test.py`

## User
- From the root of the repo, `python3 setup.py install`

Test change.	

