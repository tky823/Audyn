#!/bin/bash

set -eu
set -o pipefail

data_root="../data"

server_type="mirror"
quality="raw"

. ../../_common/parse_options.sh || exit 1;

mtg_jamendo_root="${data_root}/MTG-Jamendo/audio"

audyn-download-mtg-jamendo \
server_type="${server_type}" \
quality="${quality}" \
root="${mtg_jamendo_root}"
