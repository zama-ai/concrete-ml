run_example: three

one:
	@./train_with_docker.sh

two: one
	@python ../server/deploy_to_docker.py --only-build

three: two
	@python build_docker_client_image.py
