import os

# --- BASE DIRECTORIES ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- INPUT VARIABLES ---
DATA_DIR = os.path.join(BASE_DIR, "data")
ALLOWED_EXTENSIONS = {'pdf', 'pptx', 'csv', 'xlsx', 'xls', 'jpg', 'jpeg', 'png'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit

# --- OUTPUT VARIABLES ---
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_IMG_DIR = os.path.join(DATA_DIR, "temp_images")  # Where the Data Agent drops extracted images
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# --- MODEL CONFIGURATIONS ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
VISION_MODEL_PATH = os.path.join(MODELS_DIR, "cbam_resnet_real_anchors_v1.pth")
VISION_CONF_THRESHOLD = 0.05
VISION_ANCHOR_PRIORS = [[0.2, 0.2], [0.35, 0.25], [0.5, 0.4]]
OLLAMA_MODEL_NAME = "mistral"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- API KEYS ---
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "your_tavily_key")

# --- INITIALIZATION LOGIC ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_IMG_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
