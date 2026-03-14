# 🧠 Offline Agentic Document and Vision Explainer

> Privacy-first multimodal intelligence running fully on local hardware.

An end-to-end local AI system that ingests documents and images, performs agentic RAG for text understanding, runs custom vision inference for object detection insights, and returns one unified report in text or audio.

## ⚡ Why It Stands Out

- 🔐 100% local-first workflow for sensitive data
- 🤖 Agentic orchestration with explicit task routing
- 🧩 Text + vision fusion in one explainable output
- 🧠 Local LLM + local vector memory + local TTS
- 💻 CPU-friendly execution path

No mandatory cloud dependency for the core pipeline.

## 🚀 What It Does

1. 📤 Accepts uploads via Flask UI (PDF, PPTX, CSV, XLSX/XLS, JPG/JPEG/PNG).
2. ✂️ Extracts text and embedded images.
3. 🧭 Routes execution through a LangGraph state machine.
4. 🗂️ Chunks and embeds text into local ChromaDB.
5. 💬 Uses local Ollama model to produce structured explanation.
6. 👁️ Runs custom vision inference with configurable confidence and anchor priors.
7. 🧾 Compiles a final multimodal report.
8. 🔊 Optionally generates offline WAV narration using pyttsx3.

## 🏗️ Architecture (Agentic Flow)

```text
Upload (Flask)
  -> DataAgent (parse text + extract images)
      -> TextAgent (chunk + embed + Chroma retrieval + Ollama explanation)
      -> VisionAgent (CBAM-ResNet inference + confidence filtering + anchor priors)
  -> CompilerAgent (merge text and vision insights)
  -> Optional TTS (pyttsx3)
```

## 🧠 Workflow Nodes

- 📦 DataAgent
  - Parses the uploaded file.
  - Returns raw text and extracted image paths.
- 📝 TextAgent
  - Stores chunks in ChromaDB.
  - Retrieves relevant context.
  - Prompts local Ollama for final explanation.
  - Can add optional Tavily context only when explicitly enabled.
- 🎯 VisionAgent
  - Runs local image inference.
  - Applies confidence threshold filtering.
  - Accepts custom anchor priors.
  - Generates detection metadata and output images.
- 🧬 CompilerAgent
  - Merges text and vision outputs into final report payload.

## 👁️ Vision Model Focus

The project is wired for a custom CBAM-ResNet style detection path with anchor-aware behavior.

- 📁 Weights path: `models/cbam_resnet_no_entry_v1 .pth`
- ⚙️ Runtime controls:
  - Confidence threshold tuning
  - Anchor prior overrides from UI/API
  - Annotated detection image output in `output/`

If you have a stronger trained checkpoint, place it in `models/` and keep the configured path aligned.

## 🎨 UI Experience

The Flask frontend is intentionally styled for a modern, high-signal UX:

- 🌌 Glassmorphism-inspired dark interface with gradient accents
- 📂 Drag-and-drop upload interaction
- 🧪 Runtime controls for threshold and anchor priors
- 📶 Processing-state feedback with step-by-step status
- 🖼️ Detection image previews and structured result display
- 🔊 Audio output mode for offline speech playback

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | langgraph |
| Agents + Tooling | crewai, langchain |
| Local LLM | ollama (default: llama3) |
| Vector Memory | chromadb, langchain-chroma |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Parsing | pymupdf, python-pptx, pandas, openpyxl, xlrd |
| Vision | torch, torchvision, opencv-python, Pillow, matplotlib |
| Web App | Flask |
| Offline Audio | pyttsx3 |
| Optional Web Fallback | tavily-python |

## 📁 Project Structure

```text
task3/
├── app.py
├── config.py
├── requirements.txt
├── src/
│   ├── workflow.py
│   ├── vision_model.py
│   ├── agents/
│   │   ├── data_agent.py
│   │   └── explainer_agent.py
│   └── tools/
│       ├── parser.py
│       ├── vector_store.py
│       ├── tts_module.py
│       └── web_search.py
├── templates/
│   └── index.html
├── data/
│   ├── chroma_db/
│   └── temp_images/
├── models/
│   └── cbam_resnet_no_entry_v1 .pth
└── output/
```

## ⚙️ Local Setup

### Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin) installed and running.
- Ollama installed on the host machine.
- `llama3` must already be available locally in Ollama.

```bash
ollama pull llama3
```

- Ollama server must be running on the host before starting this app (local or Docker).

```bash
ollama serve
```

### 1) Clone and install

```bash
git clone https://github.com/Roccodevil/task3.git
cd task3
python -m venv venv
```

Activate environment:

```powershell
venv\Scripts\Activate.ps1
```

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Optional web fallback

Windows PowerShell:

```powershell
$env:TAVILY_API_KEY="your_key"
```

Linux/macOS:

```bash
export TAVILY_API_KEY="your_key"
```

### 3) Launch app

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

## 🐳 Run With Docker

This container does not run Ollama inside Docker. It connects to Ollama running on your host via `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

Make sure Ollama is running first:

```bash
ollama serve
```

And confirm `llama3` is already pulled locally:

```bash
ollama pull llama3
```

### Option A) Docker Compose (recommended)

```bash
docker compose up --build
```

To run detached:

```bash
docker compose up -d --build
```

To stop:

```bash
docker compose down
```

### Option B) Docker image + docker run

Build image:

```bash
docker build -t task3-app:latest .
```

Run container:

```bash
docker run --rm -p 5000:5000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/output:/app/output" \
  -v "${PWD}/models:/app/models" \
  --add-host=host.docker.internal:host-gateway \
  task3-app:latest
```

Open: `http://127.0.0.1:5000`

Windows PowerShell example for volume paths:

```powershell
docker run --rm -p 5000:5000 `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\output:/app/output" `
  -v "${PWD}\models:/app/models" `
  --add-host=host.docker.internal:host-gateway `
  task3-app:latest
```

## 🎮 How To Use

1. Upload a supported file.
2. Choose text or audio output.
3. Optionally enable web fallback.
4. Adjust confidence threshold and anchor priors.
5. Run analysis and review final report + detections.

## 🔌 API Surface

- `GET /` → UI page
- `POST /process` → processing endpoint

Multipart fields:

- `document` (required)
- `web_search` (optional: true/false)
- `output_format` (optional: text/audio)
- `conf_threshold` (optional: float in [0,1])
- `anchor_priors` (optional JSON list, example: [[0.2,0.2],[0.35,0.25]])

## 📈 Runtime Notes

- 🧊 First run is slower due to model and embedding initialization.
- 🗃️ ChromaDB persists in `data/chroma_db`.
- 🖼️ Extracted images are stored in `data/temp_images`.
- 🧾 Detection visuals are written to `output/`.
- 🧮 Designed for CPU-first local execution.

## 🛣️ Roadmap

- Stronger domain-tuned detection heads and richer box regression
- Streaming progress in UI
- Export-ready reporting formats
- Quantitative grounding and hallucination evaluation

## 📄 License

Add your preferred license before public release (for example, MIT).
