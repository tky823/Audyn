#!/bin/bash

set -eu
set -o pipefail

(
    cd test_parse_run_command/
    . ./train.sh
    cd -
)
