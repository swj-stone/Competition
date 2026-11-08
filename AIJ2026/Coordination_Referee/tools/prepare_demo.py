"""Restore the organizer's public demo pair from the two bundled assets."""
import argparse
import gzip
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = "8e8feb4ee284d2acd918b533921631fa905d47adf8d34881f89359964a1bcfbc"
TEST = "f005bb07868fd0f3b0953153b59871fd9f49a56f5639ce977bf617766a2387f9"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def unpack(source, target, expected):
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"{target} already exists with different content. Use a new --out directory.")
        print("verified", target)
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with gzip.GzipFile(fileobj=source) as src, tmp_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        if digest(tmp_path) != expected:
            raise ValueError(f"Unexpected decompressed content for {target.name}")
        tmp_path.replace(target)
        print("restored", target)
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "demo")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assets = ROOT / "assets"
    with zipfile.ZipFile(assets / "submission_v84b_v81_lm2.zip") as z, z.open("extra_train.csv.gz") as src:
        unpack(src, args.out / "train.csv", TRAIN)
    with (assets / "demo-test.csv.gz").open("rb") as src:
        unpack(src, args.out / "test.csv", TEST)


if __name__ == "__main__":
    main()
