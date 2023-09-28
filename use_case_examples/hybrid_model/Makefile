# Useful for jupyter notebooks
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

EXAMPLE_NAME=hybrid_model

run_example: compile_model serve_model

compile_model:
	@python compile_hybrid_llm.py

serve_model:
	@./serve.sh