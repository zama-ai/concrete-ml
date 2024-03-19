# Useful for jupyter notebooks
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

EXAMPLE_NAME=federated_learning
PYTHON_RUN=python

run_example: run_sh load_to_cml

run_sh:
	@./run.sh

load_to_cml:
	@$(PYTHON_RUN) load_to_cml.py