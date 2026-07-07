# Geographical RAG Pipeline

A highly modular, production-ready Retrieval-Augmented Generation (RAG) engine designed to query complex geographic and environmental data. This system implements an automated end-to-end pipeline: scraping structured encyclopedic articles, computing localized vector search embeddings, index-matching via FAISS, and serving contextualized configurations to open-source LLMs.

The project demonstrates production engineering patterns by exposing **three distinct execution interfaces**: a streamlined command-line terminal, an interactive browser-based dashboard, and an enterprise-standard REST API.

---

## 🚀 Key Features

* **Automated Data Ingestion:** Session-optimized web scraping using BeautifulSoup to download and sanitize localized knowledge maps.
* **Deterministic Filtering:** Supports hybrid semantic search enhanced by exact metadata filtering tags (e.g., matching text properties constrained explicitly to chosen documents).
* **Advanced Inference Guardrails:** Custom XML prompt bounding (`<context>`, `<question>`) optimized for instruction-tuned reasoning models like Llama-3.3-70B-Instruct-Turbo.
* **Robust Multi-Interface Layer:** * **CLI:** Low-latency localized console terminal for immediate diagnostic chat loops.
  * **Gradio UI:** Interactive visual web sandbox complete with runtime sliding parameters and visible context block reference mapping.
  * **FastAPI Backend:** Fully asynchronous production gateway exposing vectorized retrieval endpoints accompanied by interactive OpenAPI documentation.

---

## 🛠 Architecture & Tech Stack

* **Language:** Python 3.10+
* **Vector Store & Retrieval:** `FAISS` (Facebook AI Similarity Search) utilising continuous dense L2 Euclidean distance.
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional semantic maps).
* **Inference Model:** `deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free` (Served via Together AI infrastructure).
* **Web Frameworks:** FastAPI (Uvicorn engine) & Gradio.

---

## 📁 Repository Structure

```text
├── app/
│   ├── config.py                  # Global configurations, model definitions, and hyper-parameters
│   ├── data_utils.py              # Text cleaning, normalization, and semantic sliding-window chunking
│   ├── retrieval.py               # Vector Retreiver object handling FAISS operations and metadata queries
│   ├── generation.py              # Context assembly and Together AI inference handlers
│   ├── scrape_britannica_articles.py # Session-managed BeautifulSoup internet scraper
│   ├── build_index.py             # Main data processing pipeline to compile the vector base
│   ├── cli.py                     # Console interface script
│   ├── gui.py                     # Browser Gradio dashboard interface
│   └── main.py                    # Production FastAPI application gateway
├── .gitignore                     # Version control masking rules
├── requirements.txt               # Main dependency package manifest
└── README.md                      # Project documentation
```
---

## 💻 Installation & Environment Setup
### 1. Clone and Navigate
```
git clone https://github.com/Adel1054/Geographical-Rag-Pipeline.git
cd Geographical-Rag-Pipeline
```
### 2. Set Up a Virtual EnvironmentBash# Create environment
```
# Create environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate

# Activate environment (macOS/Linux)
source venv/bin/activate
```
### 3. Install Dependencies
Move into the code orchestration directory and pull down required packages:
```
pip install -r requirements.txt
```
### 4. Configure Your API Token
Create a ```.env``` file in the ```app/``` folder to securely link your inferencing credentials:
```
TOGETHER_API_KEY=your_actual_together_ai_token_here
```

---

## ⚙️ Running the Pipeline
Execution follows a step-by-step sequence: data gathering $\rightarrow$ vector compilation $\rightarrow$ application execution. Always run your execution commands from inside the app/ directory.
### Step 1: Ingest Local Data Text
Pull the foundational reference articles using the optimized, rate-limited web scraper:
```
python scrape_britannica_articles.py
```
This generates raw text segments under a freshly created ```../data/raw/``` path.

## Step 2: Build the Vector Database Index
Compute semantic dimensions and compile your physical search database file structures:
```
python build_index.py
```
This script uses your embedding models to generate optimized vector indices and dumps your persistent data blocks into the local ```../index/``` directory.
## Step 3: Launch Your Interface Option
Choose how you want to interact with your operational RAG backend:
### Option A: Terminal Chat Interface (CLI Mode)
Ideal for lightweight, low-overhead testing directly in your machine terminal.
```
python cli.py
```
### Option B: Full Interactive Web App (Gradio Interface)
Launches a browser sandbox UI featuring interactive parameter control.
```
python gui.py
```
Open the local loop address displayed in your terminal console (e.g., http://127.0.0.1:7860).

### Example 1: Precision Retrieval
This view demonstrates the system identifying specific geological features related to glaciations based on user queries.


<img width="1920" height="914" alt="Screenshot (379)" src="https://github.com/user-attachments/assets/af72c98e-052e-4c92-a98d-60d0e391dce0" />

### Example 2: Contextual Analysis
This view shows how the model synthesizes broader terrain descriptions while providing metadata-linked source chunks.


<img width="1920" height="933" alt="Screenshot (378)" src="https://github.com/user-attachments/assets/739b9637-5470-4bca-9034-7377346e4914" />


### Option C: Production Enterprise API Gateway (FastAPI Server)
Launches your industrial integration endpoints backed by unified server systems.
```
uvicorn main:app --reload
```
Once active, navigate your browser to ```http://127.0.0.1:8000/docs``` to test, explore, and debug live vector schemas via the embedded interactive Swagger OpenAPI panel.
