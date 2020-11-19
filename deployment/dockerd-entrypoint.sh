#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ncs --ts-config /home/model-server/config.properties
else
    eval "$@"
fi

tail -f /dev/null