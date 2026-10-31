#!/usr/bin/env python3
"""Recombine train_images.npy and test_images.npy from their .part* chunks."""
import hashlib
import os
import sys

JOBS = [
    ("train_images.npy", "train_images_recombined.npy"),
    ("test_images.npy",  "test_images_recombined.npy"),
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recombine(base, output):
    prefix = f"{base}.part"
    found = sorted(n for n in os.listdir(".") if n.startswith(prefix))
    if not found:
        print(f"[skip] no files matching '{prefix}*' in current directory")
        return False

    expected = [f"{base}.part{i:03d}" for i in range(1, len(found) + 1)]
    if found != expected:
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        sys.exit(f"Part sequence mismatch for {base}.\n  missing: {missing}\n  unexpected: {extra}")

    print(f"Recombining {len(found)} part(s) -> {output}")
    with open(output, "wb") as out:
        for name in found:
            with open(name, "rb") as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    out.write(block)
            print(f"  appended {name}")

    print(f"Done. {output}: {os.path.getsize(output):,} bytes")
    print(f"Recombined SHA-256: {sha256_of(output)}")
    return True


def main():
    any_done = False
    for base, output in JOBS:
        if recombine(base, output):
            any_done = True
        print()
    if not any_done:
        sys.exit("No parts found for any of the expected files.")


if __name__ == "__main__":
    main()