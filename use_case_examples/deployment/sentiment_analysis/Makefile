run_example: three

one:
	@python compile_with_docker.py

two: one
	@python -m concrete.ml.deployment.deploy_to_docker --only-build

three: two
	@docker build --tag cifar_client -f Dockerfile.client .
