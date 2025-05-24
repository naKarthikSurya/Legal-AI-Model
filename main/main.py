from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context,
    StorageContext,
    load_index_from_storage,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, List
import uvicorn
from functools import lru_cache
import os

app = FastAPI(
    title="RTI & Legal AI Assistant",
    description="Interactive legal assistant for RTI and Indian Law queries",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, JS, images)
app.mount("/main/static", StaticFiles(directory="main/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="main/templates")


class QueryRequest(BaseModel):
    query: str


class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    response: str
    sources: List[SourceDocument] = []


# === Load LLM === #
@lru_cache(maxsize=1)
def load_llm():
    model_name = "google/gemma-2b-it"
    # Get token from environment variable for security
    auth_token = os.getenv("HF_TOKEN", "YOUR_HUGGINGFACE_ACCESS_KEY")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir="./model/", token=auth_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="./model/",
        token=auth_token,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    system_prompt = (
        "<start_of_turn>system\n"
        "You are a helpful, honest, and unbiased legal assistant. You answer based only on Indian law and court rulings \n"
        "<end_of_turn>\n<start_of_turn>user\n"
    )

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=512,
        system_prompt=system_prompt,
        model=model,
        tokenizer=tokenizer,
    )

    embedding_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
    )

    service_context = ServiceContext.from_defaults(
        chunk_size=2048,
        llm=llm,
        embed_model=embedding_model,
    )
    set_global_service_context(service_context)

    return service_context


# === Load or Build Index === #
@lru_cache(maxsize=1)
def load_index():
    storage_path = "./storage"
    if Path(storage_path).exists():
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
    else:
        all_documents = []
        txt_dir = Path("./legaldata/final/")
        if txt_dir.exists():
            txt_loader = SimpleDirectoryReader(input_dir=str(txt_dir))
            txt_docs = txt_loader.load_data()
            all_documents.extend(txt_docs)
        index = VectorStoreIndex.from_documents(all_documents)
        index.storage_context.persist(persist_dir=storage_path)
    return index


# Dependency to get the query engine
def get_query_engine():
    load_llm()
    index = load_index()
    return index.as_query_engine()


# Route for home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route for model interface
@app.get("/model", response_class=HTMLResponse)
async def model_interface(request: Request):
    return templates.TemplateResponse("model.html", {"request": request})


# Route for about page
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


# API endpoint for queries
@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, query_engine=Depends(get_query_engine)):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        response = query_engine.query(request.query)

        # Convert source documents to the right format
        formatted_sources = []
        source_nodes = (
            response.source_nodes if hasattr(response, "source_nodes") else []
        )

        for node in source_nodes:
            formatted_sources.append(
                SourceDocument(content=node.node.text, metadata=node.node.metadata)
            )

        return QueryResponse(response=response.response, sources=formatted_sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    # Initialize directories if they don't exist
    os.makedirs("main/static", exist_ok=True)
    os.makedirs("main/templates", exist_ok=True)

    # Load the models at startup
    print("Initializing model and index...")
    try:
        load_llm()
        load_index()
        print("Model and index initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")

    # Run the API with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
