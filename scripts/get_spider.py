"""Download the official Spider 1.0 release (questions, gold SQL, SQLite DBs), pinned and verified.

The xlangai/spider parquet on the Hub has no databases, so execution accuracy needs this zip.
Source is a Hub re-host of the official zip, pinned to a commit and checked by sha256 so local
and Kaggle runs see byte-identical data.

    python scripts/get_spider.py --out data          # -> data/spider_data/{dev.json,database/...}
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path, PurePosixPath

REPO = "HAL-9001/spider-databases"
REVISION = "4a01bbac6520cd35b216db9e1724e5e1ada60aa4"
SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(zf: zipfile.ZipFile, out: Path) -> int:
    """Extract spider_data/ only (skips macOS junk); refuses absolute or '..' paths."""
    n = 0
    for info in zf.infolist():
        name = PurePosixPath(info.filename)
        if name.parts[0] != "spider_data" or name.name in {".DS_Store"}:
            continue
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"unsafe path in zip: {info.filename}")
        zf.extract(info, out)
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data"))
    a = p.parse_args()
    from huggingface_hub import hf_hub_download

    zpath = Path(hf_hub_download(REPO, "spider_data.zip", repo_type="dataset", revision=REVISION,
                                 local_dir=a.out / "raw"))
    digest = sha256(zpath)
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: {digest}")
    with zipfile.ZipFile(zpath) as zf:
        n = safe_extract(zf, a.out)
    print(f"extracted {n} entries -> {a.out / 'spider_data'}")


if __name__ == "__main__":
    main()
