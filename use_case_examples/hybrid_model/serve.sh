#!/bin/bash
uname_str=$(uname)
echo "${uname_str}"
if [[ $uname_str != "Darwin" ]]; then
  echo "Not Darwin"
  # tune the cpu-list according to the resources you want to allocate to it
  taskset --cpu-list 0-12 python serve_model.py --port 8000 --path-to-models ./compiled_models
  # No-limit
  # PATH_TO_MODELS="compiled_models" PORT=8000 python serve_model.py
else
  echo "Darwin"
  python serve_model.py --port 8000 --path-to-models ./compiled_models
fi
