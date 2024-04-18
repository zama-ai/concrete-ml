run_example: three

one:
	@./train_with_docker.sh

two: one
	@python -m concrete.ml.deployment.deploy_to_docker --only-build

three: two
	@python build_docker_client_image.py
