import torch
import time
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

# === [1] Load Gemma 2B-IT Model (4-bit CUDA) === #
print("[INFO] Loading Gemma 2B-IT model...")

model_name = "google/gemma-2b-it"
auth_token = "YOUR_HUGGINGFACE_ACCESS_KEY"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model/", token=auth_token)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./model/",
    token=auth_token,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("[INFO] Model loaded successfully.")

# === [2] LLM Wrapper === #
system_prompt = (
    "<start_of_turn>system\n"
    "You are a helpful, honest, and unbiased legal assistant. You answer based only on Indian law and court rulings.\n"
    "<end_of_turn>\n<start_of_turn>user\n"
)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    system_prompt=system_prompt,
    model=model,
    tokenizer=tokenizer,
)

# === [3] Embeddings on CUDA === #
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Loading embedding model on {device}...")
embedding_model = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )
)

# === [4] Setup Service Context === #
service_context = ServiceContext.from_defaults(
    chunk_size=2048,
    llm=llm,
    embed_model=embedding_model,
)
set_global_service_context(service_context)

# === [5] Load or Build Index === #
storage_path = "./storage"
if Path(storage_path).exists():
    print("[INFO] Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)
else:
    print("[INFO] Creating index from text files...")
    all_documents = []

    txt_dir = Path("./legaldata/final/")
    if txt_dir.exists():
        txt_loader = SimpleDirectoryReader(input_dir=str(txt_dir))
        txt_docs = txt_loader.load_data()
        print(f"[INFO] Loaded {len(txt_docs)} text files.")
        all_documents.extend(txt_docs)
    else:
        print("[WARN] No 'legaldata/final/' folder found.")

    print("[INFO] Embedding documents:")
    from tqdm import tqdm
    for _ in tqdm(all_documents):
        pass  # Just to show progress bar while embedding (actual embedding done internally)

    print(f"[INFO] Total documents for indexing: {len(all_documents)}")
    start = time.time()
    index = VectorStoreIndex.from_documents(all_documents)
    index.storage_context.persist(persist_dir=storage_path)
    print(f"[INFO] Index built in {time.time() - start:.2f} seconds.")

# === [6] Query Engine === #
query_engine = index.as_query_engine()

query = "What happens if a government department refuses RTI?"
print(f"\n[QUERY]: {query}")
response = query_engine.query(query)
print(f"\n[RESPONSE]:\n{response}")
