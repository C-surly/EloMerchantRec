#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python src/export_submission.py
python src/verify_submission.py
