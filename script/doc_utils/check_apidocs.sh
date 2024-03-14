#!/bin/bash

set -e

rm -rf docs/references/tmp.api_for_check
APIDOCS_OUTPUT=./docs/references/tmp.api_for_check make apidocs
rm -f docs/references/tmp.api_for_check/.pages
rm -f docs/references/api/.pages
diff ./docs/references/api ./docs/references/tmp.api_for_check
rm -rf docs/references/tmp.api_for_check


