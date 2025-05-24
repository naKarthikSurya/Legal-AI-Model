import streamlit as st
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

# Streamlit page config
st.set_page_config(page_title="RTI & Legal AI Assistant", page_icon="⚖️")
st.title("\u2696\ufe0f RTI & Legal Query Assistant")
st.markdown("Ask a legal question related to RTI or Indian Law:")

# Input box
query = st.text_input("What do you want to ask?")

# === Load LLM === #
@st.cache_resource(show_spinner=True)
def load_llm():
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
@st.cache_resource(show_spinner=True)
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

# Load components
with st.spinner("Loading AI model and index..."):
    load_llm()
    index = load_index()
    query_engine = index.as_query_engine()

# Run query
if query:
    with st.spinner("Thinking..."):
        response = query_engine.query(query)
    st.success("Response ready!")
    st.subheader("Answer")
    st.write(response.response)

    with st.expander("Source Documents"):
        st.write(response.get_formatted_sources())
