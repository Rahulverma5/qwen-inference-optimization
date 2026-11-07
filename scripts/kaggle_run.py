"""Run a week's notebook on Kaggle GPUs from this machine.

    python scripts/kaggle_run.py run --week 1           # upload code, push notebook, wait, fetch results
    python scripts/kaggle_run.py status --week 1
    python scripts/kaggle_run.py fetch --week 1         # results zip -> reports/week1/kaggle/
    python scripts/kaggle_run.py stage --week 1         # build upload folders locally only (dry run)
    python scripts/kaggle_run.py run --week 1 --attach models/week1_run3/sft/adapter=qwen-sql-week1-adapter
                                                        # also upload a local dir as a private dataset
    python scripts/kaggle_run.py run --week 3 --part b --kernel-source qwen-sql-week3a
                                                        # notebook week3b_kaggle.ipynb, mounting the
                                                        # output (models) of kernel qwen-sql-week3a

Code reaches Kaggle as a private dataset holding `git archive HEAD` (plus a COMMIT file), so
every run is tied to an exact commit and no public GitHub repo is needed. Uncommitted changes
are NOT uploaded: commit first. Needs the Kaggle CLI logged in (~/.kaggle).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".kaggle_stage"
CODE_SLUG = "qwen-inference-optimization-src"
ACCELERATOR = "NvidiaTeslaT4"  # Kaggle's name for "GPU T4 x2"


def sh(*args: str, check: bool = True) -> str:
    out = subprocess.run(args, capture_output=True, text=True)
    if check and out.returncode != 0:
        sys.exit(f"$ {' '.join(args)}\n{out.stdout}{out.stderr}")
    return out.stdout + out.stderr


def kaggle(*args: str, check: bool = True) -> str:
    return sh(str(Path(sys.executable).with_name("kaggle")), *args, check=check)


def username() -> str:
    for line in kaggle("config", "view").splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    sys.exit("could not read Kaggle username; run `kaggle config view`")


def kernel_slug(tag: str) -> str:
    return f"qwen-sql-week{tag}"


def stage(tag: str, user: str, attach: list[tuple[Path, str]] = (),
          kernel_sources: list[str] = ()) -> tuple[Path, Path, str]:
    if sh("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no").strip():
        print("warning: uncommitted changes to tracked files are not uploaded")
    commit = sh("git", "-C", str(ROOT), "rev-parse", "--short", "HEAD").strip()
    shutil.rmtree(STAGE, ignore_errors=True)

    code = STAGE / "code"
    code.mkdir(parents=True)
    tar = code / "src.tar"
    sh("git", "-C", str(ROOT), "archive", "--format=tar", "-o", str(tar), "HEAD")
    (code / "COMMIT").write_text(commit + "\n")

    nb_src = ROOT / "notebooks" / f"week{tag}_kaggle.ipynb"
    if not nb_src.exists():
        sys.exit(f"missing {nb_src}")
    kern = STAGE / "kernel"
    kern.mkdir()
    shutil.copy(nb_src, kern / nb_src.name)
    (kern / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{kernel_slug(tag)}",
        "title": kernel_slug(tag),
        "code_file": nb_src.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": ACCELERATOR,
        "dataset_sources": [f"{user}/{CODE_SLUG}", *(f"{user}/{slug}" for _, slug in attach)],
        "competition_sources": [],
        "kernel_sources": [f"{user}/{slug}" for slug in kernel_sources],
        "model_sources": [],
    }, indent=2))
    return code, kern, commit


def upload_dataset(folder: Path, user: str, slug: str, message: str) -> None:
    """Create or version a private dataset from `folder` and wait until Kaggle has processed it."""
    meta = folder / "dataset-metadata.json"
    if not meta.exists():
        meta.write_text(json.dumps({"title": slug, "id": f"{user}/{slug}",
                                    "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    exists = f"{user}/{slug}" in kaggle("datasets", "list", "--mine", "-s", slug, check=False)
    if exists:
        kaggle("datasets", "version", "-p", str(folder), "-m", message)
    else:
        kaggle("datasets", "create", "-p", str(folder))  # private by default
    # The kernel must not start before the new dataset version is processed.
    for _ in range(90):
        st = kaggle("datasets", "status", f"{user}/{slug}", check=False).strip().lower()
        if "ready" in st:
            print(f"dataset {slug} ready ({message})")
            return
        time.sleep(10)
    sys.exit(f"dataset {slug} not ready after 15 min: {st}")


def stage_attachment(src: Path, slug: str) -> Path:
    """Copy a local dir into the staging area so the dataset metadata file never lands in it."""
    if not src.is_dir():
        sys.exit(f"--attach source is not a directory: {src}")
    dest = STAGE / f"attach-{slug}"
    shutil.copytree(src, dest)
    return dest


def status(tag: str, user: str) -> str:
    return kaggle("kernels", "status", f"{user}/{kernel_slug(tag)}", check=False).strip()


def fetch(week: int, part: str, user: str) -> Path:
    tag = f"{week}{part}"
    dest = ROOT / "reports" / f"week{week}" / ("kaggle" + (f"_{part}" if part else ""))
    dest.mkdir(parents=True, exist_ok=True)
    # Only the small results bundle and the log; model weights stay on Kaggle as kernel output.
    kaggle("kernels", "output", f"{user}/{kernel_slug(tag)}", "-p", str(dest), "-o",
           "--file-pattern", rf"^(week{tag}_results\.zip|[^/]+\.log)$")  # top level only
    zips = list(dest.glob(f"week{tag}_results.zip"))
    if zips:
        shutil.unpack_archive(zips[0], dest / "results")
    print(f"fetched -> {dest}")
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("action", choices=["stage", "run", "status", "fetch"])
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--part", default="", help="multi-notebook weeks: a, b, ... (week3a_kaggle.ipynb)")
    p.add_argument("--no-wait", action="store_true", help="push and return without polling")
    p.add_argument("--attach", action="append", default=[], metavar="DIR=SLUG",
                   help="upload a local dir as private dataset SLUG and attach it to the run")
    p.add_argument("--kernel-source", action="append", default=[], metavar="SLUG",
                   help="mount another of your kernels' output (e.g. qwen-sql-week3a)")
    a = p.parse_args()
    user = username()
    tag = f"{a.week}{a.part}"

    if a.action == "status":
        print(status(tag, user))
        return
    if a.action == "fetch":
        fetch(a.week, a.part, user)
        return

    attach = [(Path(spec.split("=", 1)[0]), spec.split("=", 1)[1]) for spec in a.attach]
    code, kern, commit = stage(tag, user, attach, a.kernel_source)
    staged_attach = [(stage_attachment(src, slug), slug) for src, slug in attach]
    print(f"staged commit {commit} -> {STAGE}")
    if a.action == "stage":
        return

    upload_dataset(code, user, CODE_SLUG, f"commit {commit}")
    for folder, slug in staged_attach:
        upload_dataset(folder, user, slug, f"attached to commit {commit}")
    print(kaggle("kernels", "push", "-p", str(kern), "--accelerator", ACCELERATOR))
    print(f"https://www.kaggle.com/code/{user}/{kernel_slug(tag)}")
    if a.no_wait:
        return
    while True:
        st = status(tag, user)
        print(time.strftime("%H:%M:%S"), st, flush=True)
        if any(w in st.lower() for w in ("complete", "error", "cancel")):
            break
        time.sleep(120)
    fetch(a.week, a.part, user)


if __name__ == "__main__":
    main()
