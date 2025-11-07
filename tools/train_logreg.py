# Usage:
# python -m tools.train_logreg --root ".\dataset" --split train --csv train.csv --out .\out\calibrator.pkl

import os, sys, json, argparse, time, cv2, joblib
import numpy as np
import config as cfg
from pathlib import Path
from src.vqa_ollama import OllamaVQA
from src.vqa_http import HTTPVQA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# make 'src' importable when running from project root
THIS = Path(__file__).resolve()
PROJ = THIS.parents[1]
sys.path.append(str(PROJ / "src"))

# ---- MEDIA HELPERS: image OR video sampling ----
from typing import List, Tuple

KEYWORDS = [
    # smoking/vaping positives
    "cigarette","smoke","smoking","lighter","ash","puff","e-cig","vape","vaping","vapor","cloud","exhale",
    # add drinking negatives
    "bottle","water","water bottle","sports bottle","cap","unscrewing","label",
    "cup","mug","thermos","flask","straw","can","soda","juice","coffee","tea","latte","drink","drinking","sip","sipping","gulp"
]


def desc_flags(desc: str) -> list[float]:
    s = (desc or "").lower()
    return [1.0 if k in s else 0.0 for k in KEYWORDS]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS

def _read_image_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    return img if (img is not None and img.size > 0) else None

def _sample_idxs(n: int, k: int) -> List[int]:
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    # spread evenly across the clip
    return sorted(set(np.linspace(0, n - 1, k, dtype=int).tolist()))

def _read_video_cv2(path: Path, max_frames: int = 2) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release()
        return []
    frames = []
    for idx in _sample_idxs(n, max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok and f is not None and f.size > 0:
            frames.append(f)
    cap.release()
    return frames

def _read_video_imageio(path: Path, max_frames: int = 2) -> List[np.ndarray]:
    # optional fallback if OpenCV can't decode (HEVC etc.)
    try:
        import imageio.v3 as iio  # pip install imageio imageio-ffmpeg
    except Exception:
        return []
    frames = []
    try:
        meta = iio.immeta(str(path))
        n = int(meta.get("n_frames", 0)) or 0
    except Exception:
        n = 0
    # If meta failed, try to iterate anyway
    if n > 0:
        idxs = set(_sample_idxs(n, max_frames))
        for i, f in enumerate(iio.imiter(str(path))):
            if i in idxs:
                if f is not None:
                    if f.ndim == 2:  # gray
                        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    elif f.shape[2] == 4:  # RGBA -> BGR
                        f = cv2.cvtColor(f, cv2.COLOR_RGBA2BGR)
                    else:  # RGB -> BGR
                        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    frames.append(f)
            if len(frames) >= max_frames:
                break
    else:
        # fallback: just take the first few frames
        for i, f in enumerate(iio.imiter(str(path))):
            if f is None:
                continue
            if f.ndim == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            elif f.shape[2] == 4:
                f = cv2.cvtColor(f, cv2.COLOR_RGBA2BGR)
            else:
                f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            frames.append(f)
            if len(frames) >= max_frames:
                break
    return frames

def load_media_frames(path: Path, max_frames: int = 2) -> List[np.ndarray]:
    if is_image(path):
        img = _read_image_bgr(path)
        return [img] if img is not None else []
    if is_video(path):
        fr = _read_video_cv2(path, max_frames=max_frames)
        if not fr:  # try imageio-ffmpeg fallback for HEVC etc.
            fr = _read_video_imageio(path, max_frames=max_frames)
        return fr
    # Unknown extension -> try as image first
    img = _read_image_bgr(path)
    if img is not None:
        return [img]
    # then try as video
    return _read_video_cv2(path, max_frames=max_frames) or _read_video_imageio(path, max_frames=max_frames)


CLASSES = ["smoking", "vaping", "none"]  # keep as you had

def load_rows(root: Path, split: str, csv_name: str | None) -> List[Tuple[Path, str]]:
    rows: List[Tuple[Path, str]] = []

    # 1) CSV path provided and exists?
    if csv_name:
        csv_path = root / csv_name
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                cols = {name.strip().lower(): i for i, name in enumerate(header)}
                ip = cols.get("image_path", 0)
                lb = cols.get("label", 1)
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) < 2:
                        continue
                    p_str = parts[ip]
                    label = parts[lb].lower()

                    p = Path(p_str)
                    if p.is_absolute():
                        full = p
                    else:
                        # if path already starts with split name (train/val/test), join to root
                        first = p.parts[0].lower() if p.parts else ""
                        if first in {"train", "val", "test"}:
                            full = (root / p).resolve()
                        else:
                            # path is relative to the split folder
                            full = (root / split / p).resolve()

                    rows.append((full, label))
            return rows

    # 2) No CSV: scan folders root/split/<class>/*
    split_dir = root / split
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".mp4", ".mov", ".avi", ".mkv", ".webm"}
    for c in CLASSES:
        cdir = split_dir / c
        if not cdir.exists():
            continue
        for p in cdir.rglob("*"):
            if p.suffix.lower() in exts and p.is_file():
                rows.append((p.resolve(), c))
    return rows


def build_vqa_client():
    if cfg.VQA_BACKEND == "http":
        return HTTPVQA(cfg.VQA_HTTP_URL, timeout=(5, getattr(cfg, "VQA_TIMEOUT_S", 10)),
                       max_frames=getattr(cfg, "OLLAMA_MAX_FRAMES", 2))
    elif cfg.VQA_BACKEND == "ollama":
        return OllamaVQA(cfg.OLLAMA_URL, cfg.OLLAMA_MODEL,
                         timeout=(5, getattr(cfg, "VQA_TIMEOUT_S", 10)),
                         max_frames=getattr(cfg, "OLLAMA_MAX_FRAMES", 2))
    else:
        # last resort: import here to avoid hard dep if not needed
        from src.vqa_rulebased import RuleBasedVQA
        return RuleBasedVQA()


def get_probs_from_vqa(vqa_client, img_bgr):
    if hasattr(vqa_client, "classify_frames"):
        res = vqa_client.classify_frames([img_bgr])
    else:
        # fallback path (unlikely)
        tmp = Path("out") / "tmp_train.jpg"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp), img_bgr)
        res = vqa_client.classify_clip(str(tmp))
    scores = res.get("scores", {})
    return {k: float(scores.get(k, 0.0)) for k in CLASSES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root (contains train/, val/, test/)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--csv", default=None, help="(optional) CSV inside root; if omitted, scan folders")
    ap.add_argument("--out", default=str(PROJ / "out" / "calibrator.pkl"))
    ap.add_argument("--cache", default=str(PROJ / "out" / "vqa_cache_train.json"))
    args = ap.parse_args()


    root = Path(args.root)
    rows = load_rows(root, args.split, args.csv)
    if not rows:
        print("No rows found. Check CSV format (image_path,label) and paths.")
        return

    total = len(rows)
    print(f"[TRAIN] starting… 0/{total}")

    vqa_client = build_vqa_client()
    cache_path = Path(args.cache)
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    X, y = [], []
    t0 = time.time()
    for idx, (pth, label) in enumerate(rows, 1):
        key = str(pth)

        # ----- cache hit: supports both old (probs only) and new (dict with probs+desc) formats
        if key in cache:
            entry = cache[key]
            if isinstance(entry, dict) and "probs" in entry:
                probs = entry["probs"]
                desc  = entry.get("desc", "")
            else:
                probs = entry
                desc  = ""
            feat = [probs.get(c, 0.0) for c in CLASSES] + desc_flags(desc)
            X.append(feat); y.append(label)
            print(f"[TRAIN] {idx}/{total}", end="\r")
            continue

        # ----- cache miss: read media, query VQA
        frames = load_media_frames(pth, max_frames=getattr(cfg, "OLLAMA_MAX_FRAMES", 2))
        if not frames:
            print(f"[WARN] failed to read {key}")
            print(f"[TRAIN] {idx}/{total}", end="\r")
            continue

        if hasattr(vqa_client, "classify_frames"):
            res = vqa_client.classify_frames(frames)
        else:
            res = vqa_client.classify_frames([frames[0]])

        scores = res.get("scores", {})
        desc   = res.get("description", "")

        probs = {c: float(scores.get(c, 0.0)) for c in CLASSES}

        # save richer cache + build features
        cache[key] = {"probs": probs, "desc": desc}
        feat = [probs.get(c, 0.0) for c in CLASSES] + desc_flags(desc)
        X.append(feat); y.append(label)

        print(f"[TRAIN] {idx}/{total}", end="\r")

    print()  # newline after progress

    # final cache save
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))

    # Filter unknown labels
    Xf, yf = [], []
    for xi, yi in zip(X, y):
        if yi in CLASSES:
            Xf.append(xi)
            yf.append(yi)
    X = np.asarray(Xf, dtype=np.float32)
    y = np.asarray(yf, dtype=str)

    if len(np.unique(y)) < 2:
        print("Not enough label variety to train. Need at least 2 classes.")
        return

    # Build a small pipeline: standardize -> multinomial logistic
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logreg", LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000, C=2.0))
    ])
    clf.fit(X, y)

    # Save as an interoperable dict + joblib for robustness
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"type": "logreg", "classes": CLASSES, "keywords": KEYWORDS, "sk_pipeline": clf}, out_path)
    print(f"[OK] Trained calibrator saved to {out_path} in {time.time()-t0:.1f}s")
    print("Class counts:", {c: int((y==c).sum()) for c in CLASSES})


if __name__ == "__main__":
    main()
