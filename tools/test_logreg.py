# Usage:
# python tools.test_logreg --root ".\dataset" --split test --csv test.csv --cal .\out\calibrator.pkl --out .\out\eval_test.csv

import os, sys, json, argparse, time
from pathlib import Path
import numpy as np
import cv2

THIS = Path(__file__).resolve()
PROJ = THIS.parents[1]
sys.path.append(str(PROJ / "src"))

import joblib
import config as cfg
from src.vqa_ollama import OllamaVQA
from src.vqa_http import HTTPVQA

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ---- MEDIA HELPERS: image OR video sampling ----
from typing import List, Tuple

KEYWORDS = [
    # smoking/vaping positives
    "cigarette","smoke","smoking","lighter","ash","puff","e-cig","vape","vaping","vapor","cloud","exhale",
    # add drinking negatives
    "bottle","water","water bottle","sports bottle","cap","unscrewing","label",
    "cup","mug","thermos","flask","straw","can","soda","juice","coffee","tea","latte","drink","drinking","sip","sipping","gulp"
]


def desc_flags(desc: str, keywords: list[str]) -> list[float]:
    s = (desc or "").lower()
    return [1.0 if k in s else 0.0 for k in keywords]

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
        from src.vqa_rulebased import RuleBasedVQA
        return RuleBasedVQA()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="Requery VQA even if cache exists or if cache lacks description")
    ap.add_argument("--root", required=True, help="dataset root (contains train/, val/, test/)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--csv", default=None, help="(optional) CSV inside root; if omitted, scan folders")
    ap.add_argument("--cal", default=str(PROJ / "out" / "calibrator.pkl"))
    ap.add_argument("--out", default=str(PROJ / "out" / "eval_test.csv"))
    ap.add_argument("--cache", default=str(PROJ / "out" / "vqa_cache_test.json"))
    args = ap.parse_args()

    rows = load_rows(Path(args.root), args.split, args.csv)
    if not rows:
        print("No rows found. Check CSV paths.")
        return

    total = len(rows)
    print(f"[TEST] starting… 0/{total}")

    # load calibrator
    cal = joblib.load(args.cal)
    if not isinstance(cal, dict) or cal.get("type") != "logreg":
        print(f"[WARN] {args.cal} is not a logreg calibrator; exiting.")
        return
    clf = cal["sk_pipeline"]
    KEYWORDS = cal.get("keywords", [])

    # VQA client
    vqa_client = build_vqa_client()

    # cache to avoid requery
    cache_path = Path(args.cache)
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            pass

    X, y, paths, descs = [], [], [], []
    vqa_answers, vqa_raw_probs = [], []  # NEW: raw VQA answer + raw probs (dict per sample)

    for idx, (pth, label) in enumerate(rows, 1):
        key = str(pth)

        # Decide if we must requery
        need_query = args.refresh or (key not in cache) \
            or (isinstance(cache.get(key), dict) and "desc" not in cache[key])  # old cache format

        if not need_query:
            # ---- cache hit: support old/new formats
            entry = cache[key]
            if isinstance(entry, dict) and "probs" in entry:
                probs = entry["probs"]
                desc  = entry.get("desc", "")
                ans   = entry.get("answer", "")           # may be missing in older caches
                rawp  = entry.get("raw_probs", probs)     # fallback to probs if raw missing
            else:
                # old format: only probs (force requery to get description & answer next time)
                need_query = True

        if need_query:
            frames = load_media_frames(pth, max_frames=getattr(cfg, "OLLAMA_MAX_FRAMES", 2))
            if not frames:
                print(f"[WARN] read fail {key}")
                print(f"[TEST] {idx}/{total}", end="\r")
                continue

            if hasattr(vqa_client, "classify_frames"):
                res = vqa_client.classify_frames(frames)
            else:
                res = vqa_client.classify_frames([frames[0]])

            scores = res.get("scores", {}) or {}
            desc   = res.get("description", "") or ""
            ans    = res.get("answer", "") or ""
            # Keep original VQA probabilities separately
            rawp   = {c: float(scores.get(c, 0.0)) for c in CLASSES}
            probs  = rawp  # features are based on the same probs as before

            # store richer cache
            cache[key] = {"probs": probs, "desc": desc, "answer": ans, "raw_probs": rawp}

        # features: probs + keyword flags
        feat = [probs.get(c, 0.0) for c in CLASSES] + desc_flags(desc, KEYWORDS)
        X.append(feat); y.append(label); paths.append(key); descs.append(desc)
        vqa_answers.append(ans); vqa_raw_probs.append(rawp)

        print(f"[TEST] {idx}/{total}", end="\r")


    print()  # newline after progress

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=str)

    # predict calibrated classes
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)  # probabilities in clf.classes_ order

    # Map proba to CLASSES order for saving and ROC
    class_index = {c: i for i, c in enumerate(clf.classes_)}
    probs_out = []
    y_prob_aligned = np.zeros((len(y_prob), len(CLASSES)), dtype=np.float32)
    for r, row in enumerate(y_prob):
        aligned = []
        for i, c in enumerate(CLASSES):
            if c in class_index:
                y_prob_aligned[r, i] = float(row[class_index[c]])
            aligned.append(float(row[class_index[c]]) if c in class_index else 0.0)
        probs_out.append({c: v for c, v in zip(CLASSES, aligned)})

    # --- AUC-ROC evaluation ---
    try:
        y_bin = label_binarize(y, classes=CLASSES)  # shape (N, C)
        # Guard against degenerate classes (all 0 or all 1)
        valid_cols = []
        for i in range(len(CLASSES)):
            pos = int(y_bin[:, i].sum())
            neg = int(len(y_bin) - pos)
            if pos > 0 and neg > 0:
                valid_cols.append(i)

        if valid_cols:
            auc_macro = roc_auc_score(y_bin[:, valid_cols], y_prob_aligned[:, valid_cols],
                                      average="macro", multi_class="ovr")
            print(f"AUC-ROC (macro-average over valid classes): {auc_macro:.4f}")

            # Plot ROC curves per valid class
            plt.figure(figsize=(6, 5))
            for i in valid_cols:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_aligned[:, i])
                auc_i = roc_auc_score(y_bin[:, i], y_prob_aligned[:, i])
                plt.plot(fpr, tpr, label=f"{CLASSES[i]} (AUC={auc_i:.3f})")
            plt.plot([0, 1], [0, 1], "k--", lw=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves per Class")
            plt.legend(loc="lower right")
            plt.tight_layout()

            # Save and show
            outp = Path(args.out)
            roc_path = outp.with_name(outp.stem + "_roc.png")
            plt.savefig(roc_path)
            print(f"[OK] Saved ROC plot to {roc_path}")
            plt.show()
        else:
            print("[WARN] ROC/AUC skipped: not enough class variety (need positives and negatives per class).")
    except Exception as e:
        print("[WARN] Could not compute ROC/AUC:", e)

    # metrics
    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred, labels=CLASSES)
    print(f"Accuracy: {acc:.4f}  |  F1-macro: {f1m:.4f}")
    print("Confusion matrix (rows=gt, cols=pred):")
    print(CLASSES)
    print(cm)
    print("Report:\n", classification_report(y, y_pred, labels=CLASSES))

    # save per-sample results
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        f.write("image_path,gt,"
                "vqa_answer,raw_psmoking,raw_pvaping,raw_pnone,"
                "pred,psmoking,pvaping,pnone,description\n")
        for p, gt, pr, prb, desc, ans, rawp in zip(paths, y, y_pred, probs_out, descs, vqa_answers, vqa_raw_probs):
            safe_desc = (desc or "").replace('"', "''")
            f.write(f"{p},{gt},{ans},"
                    f"{rawp.get('smoking',0.0):.4f},{rawp.get('vaping',0.0):.4f},{rawp.get('none',0.0):.4f},"
                    f"{pr},{prb['smoking']:.4f},{prb['vaping']:.4f},{prb['none']:.4f},\"{safe_desc}\"\n")
    print(f"[OK] Saved per-sample eval to {outp}")



if __name__ == "__main__":
    main()
