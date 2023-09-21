# Useful for jupyter notebooks
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

EXAMPLE_NAME=cifar_brevitas_finetuning
JUPYTER_RUN=jupyter nbconvert --to notebook --inplace --execute
TIME_NB="${USE_CASE_DIR}/time_notebook_execution.sh"

run_example: four

one:
	@$(TIME_NB) FromImageNetToCifar.ipynb

two: one
	@$(TIME_NB) CifarQuantizationAwareTraining.ipynb

three: two
	@$(TIME_NB) CifarInFhe.ipynb

four: three
	@$(TIME_NB) CifarInFheWithSmallerAccumulators.ipynb

