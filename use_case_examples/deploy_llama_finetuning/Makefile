SOURCE_DIR=./compiled_models
MATCH_PATTERN=meta-llama/inference_model.base_model.model.*
ARCHIVE_DIR=./deployment/compiled_models.tar.gz

.PHONY: archive_remote_layers

archive_remote_layers:
	@echo "The size before archiving:"
	du -sh $(SOURCE_DIR)

	@echo "Creating archive at $(ARCHIVE_DIR)..."
	mkdir -p $(dir $(ARCHIVE_DIR))
	time tar --disable-copyfile -czf $(ARCHIVE_DIR) $(SOURCE_DIR)/$(MATCH_PATTERN)

	@echo "The size after archiving:"
	du -sh $(ARCHIVE_DIR)
