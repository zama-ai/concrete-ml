# Useful for jupyter notebooks
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

EXAMPLE_NAME=sentiment_analysis_with_transformers
JUPYTER_RUN=jupyter nbconvert --to notebook --inplace --execute
TIME_NB="${USE_CASE_DIR}/time_notebook_execution.sh"

run_example: one

one:
	@$(TIME_NB) SentimentClassification.ipynb
