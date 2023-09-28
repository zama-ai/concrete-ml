# Useful for jupyter notebooks
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

EXAMPLE_NAME=federated_learning
PYTHON_RUN=python

run_example: run_sh load_to_cml

run_sh:
	@echo "Running federated learning training script..."
	@./run.sh

load_to_cml:
	@echo "Loading and compiling the model with Concrete ML..."
	@$(PYTHON_RUN) load_to_cml.py
