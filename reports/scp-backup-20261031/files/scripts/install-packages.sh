#!/usr/bin/env bash
set -euo pipefail
source /home/dev/assumeengage/config/env.sh
/home/dev/assumeengage/envs/jev/bin/uv pip install \
  --python /home/dev/assumeengage/envs/jev/bin/python \
  --index-url "$UV_INDEX_URL" \
  --index-strategy unsafe-best-match \
  -r /home/dev/assumeengage/config/requirements.txt
