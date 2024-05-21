run_example: three

one:
	@python compile_with_docker.py

two: one
	@python ../server/deploy_to_docker.py --only-build

three: two
	@docker build --tag cifar_client -f Dockerfile.client .
