SOURCE_DIR := compiled_models
ARCHIVE_DIR := deployment/compiled_models.tar.gz
MATCH_PATTERN := $(wildcard $(SOURCE_DIR)/meta-llama/inference_model.base_model.model.*)
FILES_REL := $(patsubst $(SOURCE_DIR)/%,%,$(MATCH_PATTERN))

.PHONY: archive_remote_layers
archive_remote_layers:
	@echo "Size before archiving..."
	du -sh $(SOURCE_DIR)

	@echo "Creating archive at $(ARCHIVE_DIR)..."
	mkdir -p $(dir $(ARCHIVE_DIR))
	time tar --disable-copyfile -C $(SOURCE_DIR) -czf $(ARCHIVE_DIR) $(FILES_REL)

	@echo "Size after archiving..."
	du -sh $(ARCHIVE_DIR)
