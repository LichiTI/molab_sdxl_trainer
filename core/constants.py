"""
Lulynx Core Constants
"""

# === Path Defaults ===
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_OUTPUT_NAME = "my_lora"
DEFAULT_LOG_DIR = "./output/logs"
DEFAULT_QUARANTINE_DIR = "./quarantine"

# === Architecture Defaults ===
DEFAULT_NETWORK_DIM = 32
DEFAULT_NETWORK_ALPHA = 16
DEFAULT_DEVICE = "cuda"

# === Optimizer & LR Defaults ===
DEFAULT_LR = 0.0001
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EMA_DECAY = 0.999

# === Performance & Precision ===
DEFAULT_MIXED_PRECISION = "bf16"
DEFAULT_SAVE_PRECISION = "fp16"

# === Training Loop Defaults ===
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEED = 1337
DEFAULT_RESOLUTION = 1024
DEFAULT_CHECKPOINT_KEEP_LAST = 0 # 0 means keep all

# === Auditor & Monitoring (V10.0) ===
DEFAULT_AUDITOR_INTERVAL = 50
PROJECTION_SEED = 42
DEFAULT_VRAM_MARGIN_MB = 200
VRAM_THRESHOLD_STOP = 0.98
VRAM_THRESHOLD_LITE = 0.95

# === Dynamic Pruning ===
DEFAULT_PRUNE_THRESHOLD = 0.05
DEFAULT_MIN_RANK = 8

# === Training Pilot ===
PILOT_EMA_ALPHA = 0.1
PILOT_EMA_BASELINE_SPIKE = 1.5
PILOT_EMA_BASELINE_DROP = 0.5

# === LISA ===
DEFAULT_LISA_ACTIVE_RATIO = 0.2
DEFAULT_LISA_INTERVAL = 1

# === PiSSA & SmartRank ===
DEFAULT_PISSA_ENABLED = False
DEFAULT_SMART_RANK_INTERVAL = 50
DEFAULT_SMART_RANK_MIN = 4
DEFAULT_SMART_RANK_MAX = 128

# === File Naming & Extensions ===
FILENAME_MODEL_TEMPLATE = "{output_name}-{epoch:06d}"
FILENAME_STATE_TEMPLATE = "{output_name}-{epoch:06d}-state"
EXT_SAFETENSORS = ".safetensors"
EXT_PT = ".pt"

# === Security ===
import os
# Default to a dev key if not set. In production/release, this should be random or forced.
# [SECURITY] Ensure this key is set in production environment variables
REAPP_API_KEY = os.getenv("REAPP_API_KEY")
if not REAPP_API_KEY:
    # Generate a temporary random key for session safety if not provided
    import secrets
    REAPP_API_KEY = secrets.token_hex(32)
