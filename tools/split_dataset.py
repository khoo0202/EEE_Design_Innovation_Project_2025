# Usage examples:
#   # copy files into a new dataset folder
#   python tools/split_dataset.py --src "..\\dataset_sorted" --out "..\\dataset" --copy
#   # move files instead of copy (faster, destructive)
#   python tools/split_dataset.py --src "..\\dataset_sorted" --out "..\\dataset" --move
#   # custom ratios + deterministic split
#   python tools/split_dataset.py --src "..\\dataset_sorted" --out "..\\dataset" --copy --train 0.7 --val 0.2 --test 0.1 --seed 42


import argparse
import random
import shutil
from pathlib import Path
from __future__ import annotations

# treat these as valid media types (images + videos)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif",
        ".mp4", ".mov", ".avi", ".mkv", ".webm"}

CLASSES_DEFAULT = ["smoking", "vaping", "none"]

def list_media(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in EXTS and p.is_file()])

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = dst
    k = 1
    while out.exists():
        out = dst.with_name(f"{dst.stem} ({k}){dst.suffix}")
        k += 1
    shutil.copy2(src, out)
    return out

def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = dst
    k = 1
    while out.exists():
        out = dst.with_name(f"{dst.stem} ({k}){dst.suffix}")
        k += 1
    shutil.move(str(src), str(out))
    return out

def write_manifest(csv_path: Path, rows: list[tuple[str, str]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image_path,label\n")
        for rel, label in rows:
            f.write(f"{rel},{label}\n")

def split_indices(n: int, r_train: float, r_val: float, r_test: float, seed: int):
    # normalize ratios just in case
    s = r_train + r_val + r_test
    r_train, r_val, r_test = r_train/s, r_val/s, r_test/s
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_train = int(round(n * r_train))
    n_val   = int(round(n * r_val))
    # ensure totals add up to n
    n_test  = n - n_train - n_val
    i_train = set(idxs[:n_train])
    i_val   = set(idxs[n_train:n_train+n_val])
    i_test  = set(idxs[n_train+n_val:])
    return i_train, i_val, i_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source root with class folders (e.g., dataset_sorted)")
    ap.add_argument("--out", required=True, help="Output root to create train/val/test structure")
    ap.add_argument("--classes", nargs="*", default=CLASSES_DEFAULT, help="Class folder names")
    ap.add_argument("--train", type=float, default=0.8, help="Train ratio")
    ap.add_argument("--val",   type=float, default=0.1, help="Val ratio")
    ap.add_argument("--test",  type=float, default=0.1, help="Test ratio")
    ap.add_argument("--seed",  type=int, default=1337, help="Random seed")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--copy", action="store_true", help="Copy files (default)")
    g.add_argument("--move", action="store_true", help="Move files (destructive)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    out_train = out / "train"
    out_val   = out / "val"
    out_test  = out / "test"

    # prepare destination dirs
    for split_root in (out_train, out_val, out_test):
        for c in args.classes:
            (split_root / c).mkdir(parents=True, exist_ok=True)

    use_move = args.move and not args.copy

    # manifests (relative paths from out root)
    rows_train, rows_val, rows_test = [], [], []

    for c in args.classes:
        class_src = src / c
        if not class_src.exists():
            print(f"[WARN] missing class folder: {class_src}")
            continue

        files = [p for p in class_src.iterdir() if p.suffix.lower() in EXTS and p.is_file()]
        files.sort()
        n = len(files)
        if n == 0:
            print(f"[INFO] no files in {class_src}")
            continue

        i_train, i_val, i_test = split_indices(n, args.train, args.val, args.test, args.seed)

        print(f"[{c}] total={n}  train={len(i_train)}  val={len(i_val)}  test={len(i_test)}")

        for i, p in enumerate(files):
            if i in i_train:
                dst = out_train / c / p.name
                newp = safe_move(p, dst) if use_move else safe_copy(p, dst)
                rel = newp.relative_to(out).as_posix()
                rows_train.append((rel, c))
            elif i in i_val:
                dst = out_val / c / p.name
                newp = safe_move(p, dst) if use_move else safe_copy(p, dst)
                rel = newp.relative_to(out).as_posix()
                rows_val.append((rel, c))
            else:
                dst = out_test / c / p.name
                newp = safe_move(p, dst) if use_move else safe_copy(p, dst)
                rel = newp.relative_to(out).as_posix()
                rows_test.append((rel, c))

    # write CSVs
    write_manifest(out / "train.csv", rows_train)
    write_manifest(out / "val.csv", rows_val)
    write_manifest(out / "test.csv", rows_test)

    print(f"[OK] Wrote manifests:\n  {out/'train.csv'}\n  {out/'val.csv'}\n  {out/'test.csv'}")
    print(f"[DONE] Output dataset at: {out}")

if __name__ == "__main__":
    main()
