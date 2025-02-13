import os

# 🔧 Disable Streamlit’s file watcher & fix PyTorch issues
os.environ["STREAMLIT_WATCHDOG"] = "false"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["TORCH_DONT_LOAD_FALLBACK_MODULES"] = "1"
os.environ["PYTORCH_NO_CUDA"] = "1"  # 🚀 Force PyTorch to avoid CUDA issues

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64

# ✅ MODEL AND TOKENIZER LOADING
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
device = "cpu"  # 🚀 Explicitly set to CPU
tokenizer = T5Tokenizer.from_pretrained(checkpoint, legacy=False)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, torch_dtype=torch.float32, device_map=None
).to(device)

# ✅ FILE LOADER AND PREPROCESSING
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)

    final_texts = ""
    for text in texts:
        final_texts += text.page_content + " "

    return final_texts

# ✅ SUMMARIZATION FUNCTION
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50,
        device=-1,  # 🚀 Force CPU usage
    )

    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    
    return result[0]['summary_text']

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# ✅ STREAMLIT APP
st.set_page_config(layout="wide", page_title="Summarization App")

def main():
    st.title("Document Summarization App using Language Model")
    uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            
            filepath = os.path.join("data", uploaded_file.name)
            os.makedirs("data", exist_ok=True)
            
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info("Uploaded PDF File")
                displayPDF(filepath)

            with col2:
                st.info("Summarization is below")
                summary = llm_pipeline(filepath)
                st.success(summary)

if __name__ == "__main__":
    main()
