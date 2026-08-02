#!/usr/bin/env bash
# Download the Middlebury optical flow "other" benchmark set (~25 MB) into data/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
cd data
[ -d other-data ]    || { curl -LO https://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip \
                          && unzip -oq other-color-twoframes.zip && rm -f other-color-twoframes.zip; }
[ -d other-gt-flow ] || { curl -LO https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip \
                          && unzip -oq other-gt-flow.zip && rm -f other-gt-flow.zip; }
echo "Done. Inventory:"
cd .. && exec uv run python benchmark_data.py
