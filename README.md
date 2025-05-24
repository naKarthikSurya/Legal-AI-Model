
# ⚖️ Legal AI Assistant

A powerful, privacy-preserving legal information retrieval system designed specifically for Indian law. This system uses a locally deployed **Gemma 2B-IT** transformer model, optimized for consumer-grade machines with **4-bit quantization**, combined with **LlamaIndex** and **semantic search** for accurate, real-time, and context-aware answers to legal queries.

> 🔐 All data is processed locally — no cloud APIs — ensuring 100% user data privacy.

---

## Features

- Semantic search over case judgments and laws
- RAG (Retrieval-Augmented Generation) pipeline
- LLM-powered answers using Gemma 2B-IT (4-bit)
- Runs locally on machines with 8GB VRAM (CUDA)
- Optimized for Indian laws and legal terms
- FastAPI-based backend with Streamlit UI
- Document citation + explainable highlights
- Works with case judgments + IPC/RTI/Environmental Acts

---

## Directory Structure

```plaintext
LEGAL-AI-MODEL/
│
├── legaldata/
│   ├── raw/                    # Raw PDFs: case_judgements, laws_acts
│   └── final/                  # Combined TXT file: Legal_corpus.txt
│
├── main/
│   ├── static/                 # Frontend CSS, JS
│   ├── templates/              # HTML Templates
│   └── main.py                 # FastAPI routing + frontend logic
│
├── model/                      # Gemma model & tokenizer
│
├── storage/                    # LlamaIndex vector stores
│   ├── docstore.json
│   ├── graph_store.json
│   ├── index_store.json
│   └── vector_store.json
│
├── app.py                      # Streamlit front-end
├── data.py                     # Preprocessing logic
├── cuda.py                     # CUDA setup & model loading
├── legal_system.py             # Core RAG + generation pipeline
├── requirements.txt
└── README.md                   # Project documentation
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/naKarthikSurya/Legal-AI-Model.git
cd Legal-AI-Model.git
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the FastAPI Server

```bash
python main/main.py
```

Visit: `http://localhost:8000`

### 4. Run Streamlit (Optional UI)

```bash
streamlit run app.py
```

---

## Model Pipeline

- **LLM**: `google/gemma-2b-it` (4-bit quantized)
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Indexing & Search**: LlamaIndex with semantic similarity
- **Retrieval**: Top-k dense vector search
- **Answer Generation**: Contextual response using the transformer

---

## Legal Dataset

- RTI Act, IPC, Environmental Acts, etc.
- 4000+ Case Judgments (manually scraped from Indian Kanoon)
- Combined into `Legal_corpus.txt`

---


## Privacy First

- No internet-based API calls
- Everything runs offline (including LLM inference)
- Suitable for private organizations or court-integrated systems

---

## Target Users

- Law Students & Faculty
- Citizens seeking legal remedies
- Legal Researchers & NGOs
- Government/legal tech innovators

---

## Tech Stack

- **FastAPI** – Backend Framework
- **Streamlit** – Interactive Frontend
- **Gemma 2B-IT** – LLM (4-bit, local inference)
- **LlamaIndex** – RAG + Indexing
- **HuggingFace Transformers** – Model hub integration
- **SentenceTransformers** – Embeddings

---

## 📜 License

Licensed under the MIT License.

---

## 🙏 Acknowledgements

- Google’s Gemma Model
- Meta’s LLaMA3 Architecture
- LlamaIndex by Jerry Liu
- HuggingFace & SentenceTransformers
- Indian Judiciary Open Legal Datasets

---

> “Justice must not only be done, but must also be accessible — Legal AI bridges the gap.”
