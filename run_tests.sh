#!/usr/bin/env bash
# Run the full test suite — no YOLO weights required.
set -e
cd "$(dirname "$0")"
echo "=== Unit: compliance ==="
python3 tests/test_compliance.py
echo
echo "=== Unit: smoothing ==="
python3 tests/test_smoothing.py
echo
echo "=== Integration: full video pipeline ==="
python3 tests/test_integration.py
echo
echo "All tests passed."
