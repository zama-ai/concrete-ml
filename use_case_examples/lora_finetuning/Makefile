# Useful for jupyter notebooks
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

TIME_NB="${USE_CASE_DIR}/time_notebook_execution.sh"

run_example: one two

one:
	@$(TIME_NB) GPT2FineTuneHybrid.ipynb

two:
	@$(TIME_NB) LLamaFineTuning.ipynb