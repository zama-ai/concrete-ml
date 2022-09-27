#!/bin/bash

set -e

rm -rf docs/developer-guide/tmp.api_for_check
APIDOCS_OUTPUT=./docs/developer-guide/tmp.api_for_check make apidocs
rm -f docs/developer-guide/tmp.api_for_check/.pages
rm -f docs/developer-guide/api/.pages
diff ./docs/developer-guide/api ./docs/developer-guide/tmp.api_for_check
rm -rf docs/developer-guide/tmp.api_for_check


