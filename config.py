# config.py
RTSP_OR_FILE = "video/ScreenRec_251028_75578.mp4" 
# ^^^ "rtsp://admin:Chitomin2000@115.66.143.39:10554/Streaming/Channels/101" # 0 for webcam, or "rtsp://user:pass@host:554/Streaming/Channels/101" or a video path
YOLO_WEIGHTS = "yolo11n.pt"
DEVICE = "cuda"  # "cuda" or "cpu"

# --- Hugging Face VQA ---
HF_TOKEN = "hf_***YOUR_TOKEN***"  # create at https://huggingface.co/settings/tokens
HF_VQA_MODEL = "dandelin/vilt-b32-finetuned-vqa"

# --- VQA settings ---
VQA_BACKEND = "http"   # "ollama" | "http" | "rulebased"
VQA_HTTP_URL = "http://127.0.0.1:8012/vqa"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5vl:3b"
OLLAMA_MAX_FRAMES = 2

ASYNC_VQA = True
VQA_MAX_CONCURRENCY = 2
VQA_QUEUE_SIZE = 32
VQA_TIMEOUT_S = 10

# --- snapshot and pipeline timing ---
SNAPSHOT_SECS = 5
ASSUMED_FPS = 60
SEND_INTERVAL_S = 6

# --- detector and tracker thresholds ---
CONF_THRESH = 0.45
IOU_THRESH = 0.6
YOLO_IMGSZ = 640
TRACK_IOU = 0.4
TRACK_MAX_AGE = 12
TRACK_MIN_HITS = 2
INIT_MAX_AGE = 8
SMOOTH_ALPHA = 0.7

# --- re-identification / re-entry ---
REENTRY_MAX_AGE = 300
REENTRY_APP_THRESH = 0.9
REENTRY_MIN_HITS = 5
REID_GALLERY_SIZE = 5

# --- outputs ---
RESULTS_CSV = "out/results.csv"
SNAP_DIR = "out/snaps"
CALIBRATOR_PATH = "out/calibrator.pkl"

# --- reid model (ONNX) ---
APPEARANCE_MODEL_PATH = "data/reid/osnet_x0_25_msmt17.onnx"

# Weighting of spatial vs visual match when deciding if it's the same person
REID_IOU_WEIGHT = 0.8      # trust bbox overlap
REID_SIM_WEIGHT = 0.2      # trust visual embedding
# these two should add up to ~1.0

# Minimum appearance similarity required if IOU is weak
REID_MIN_SIM = 0.45        # raise towards 0.45 if IDs still jump between people

# ---Database ---
DB_PATH="./out/violations.db"
EVIDENCE_ROOT="./out/evidence"

# --- BEHAVIOURAL MODEL ---
BEHAVIOR_MODEL_PATH = "best_model_weightage.pt"
