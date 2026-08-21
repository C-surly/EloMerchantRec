#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

from export_submission import DEFAULT_OUTPUT, build_bytes

EXPECTED_SHA256 = "e708c224d1c28c790027d0e3e3b01196885b0f39272040edd0b952d51ad117e1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    if DEFAULT_OUTPUT.exists():
        payload = DEFAULT_OUTPUT.read_bytes()
    else:
        payload = build_bytes()
    digest = sha256_bytes(payload)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"sha256 mismatch: {digest}")
    print("OK 3.59428")


if __name__ == "__main__":
    main()
