#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fetch the Hiera-Hand checkpoints into checkpoints/.

The Zenodo record is one 5.1 GB archive (dataset + checkpoints). This streams
it, keeps only the requested checkpoints, and stops early: they sit near the
start, so only ~0.4 GB (manipulation) or ~0.8 GB (both) is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
URL = "https://zenodo.org/records/14958923/files/childplay_hand.tar.gz?download=1"
MEMBER = "childplay_hand/checkpoints/{}"
SHA256 = {
    "hiera_manipulation_hand.ckpt": "1968e6e51d114abf052d67f09c0208f72a30ad557c68a312747ff5905f09586c",
    "hiera_object_hand.ckpt": "e7869def3c8628f644fdb72a19b560f17648fd948ce02222f8f977df45c1f951",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Progress:
    """File-like wrapper that reports how many compressed bytes were read."""

    def __init__(self, raw, total):
        self.raw, self.bar = raw, tqdm(total=total, unit="B", unit_scale=True, desc="archive")

    def read(self, size=-1):
        data = self.raw.read(size)
        self.bar.update(len(data))
        return data


def extract(source, wanted, out_dir):
    """Stream the tar.gz and write only `wanted` checkpoint files."""
    remaining = set(wanted)
    with tarfile.open(fileobj=source, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.name != MEMBER.format(name) or name not in remaining:
                continue
            partial = out_dir / f"{name}.part"
            with tar.extractfile(member) as src, open(partial, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            if sha256(partial) != SHA256[name]:
                partial.unlink()
                raise RuntimeError(f"{name}: checksum mismatch; try again")
            partial.rename(out_dir / name)
            print(f"Saved {out_dir / name}")
            remaining.discard(name)
            if not remaining:
                return  # stop before reading the rest of the archive
    raise RuntimeError(f"Not found in archive: {', '.join(sorted(remaining))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("manipulation", "object", "all"), default="manipulation")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded childplay_hand.tar.gz")
    args = parser.parse_args()

    out_dir = ROOT / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    tasks = ("manipulation", "object") if args.task == "all" else (args.task,)
    wanted = []
    for task in tasks:
        name = f"hiera_{task}_hand.ckpt"
        if (out_dir / name).is_file() and sha256(out_dir / name) == SHA256[name]:
            print(f"Already present: {out_dir / name}")
        else:
            wanted.append(name)
    if not wanted:
        return

    if args.archive:
        with open(args.archive, "rb") as handle:
            source = Progress(handle, args.archive.stat().st_size)
            extract(source, wanted, out_dir)
    else:
        with urllib.request.urlopen(URL) as response:
            source = Progress(response, int(response.headers.get("Content-Length", 0)) or None)
            extract(source, wanted, out_dir)
    source.bar.close()


if __name__ == "__main__":
    main()
