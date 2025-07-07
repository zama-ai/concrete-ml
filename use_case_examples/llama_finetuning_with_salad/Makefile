SOURCE_DIR=./compiled_models
SOURCE_COMPRESSED_DIR=./deployment_float/compiled_models
ARCHIVE_DIR=./deployment_float/compiled_models.tar.gz

.PHONY: extract_remote_layers archive_remote_layers

extract_remote_layers:
	mkdir -p $(dir $(SOURCE_COMPRESSED_DIR))
	python extract_remote_layers.py --dest_root $(dir $(SOURCE_COMPRESSED_DIR))

archive_remote_layers:
	@echo "The size before RETRAIN:"
	du -sh $(SOURCE_DIR)

	@echo "The size after archiving:"
	du -sh $(dir $(SOURCE_COMPRESSED_DIR))

	# tar -czf <archive> -C <directory> <what to include>
	time tar --disable-copyfile -czf $(ARCHIVE_DIR) -C $(SOURCE_COMPRESSED_DIR) .

	@echo "The size after archiving:"
	du -sh $(ARCHIVE_DIR)
