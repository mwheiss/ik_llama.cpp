#!/bin/bash

set -eu

python3 "$(dirname "$0")/prepare_fixtures.py"

if [ $# -lt 1 ]
then
    # Start @llama.cpp scenario
    behave --summary --stop --no-capture --exclude 'issues|wrong_usages|passkey' --tags llama.cpp
else
    behave "$@"
fi
