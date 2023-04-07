# Useful for jupyter notebooks
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

EXAMPLE_NAME=cifar_brevitas_finetuning
JUPYTER_RUN=jupyter nbconvert --to notebook --inplace --execute

run_example: three

one:
	@python compile_with_docker.py

two: one
	@python -m concrete.ml.deployment.deploy_to_docker --only-build

three: two
	@python build_docker_client_image.py
