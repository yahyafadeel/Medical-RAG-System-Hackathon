from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import config

def load_pdfs(data_dir):
    """
    Function to load PDFs and normalize their metadata for citation.
    """
    pages = []
    # هندور على كل ملفات الـ PDF اللي في فولدر الداتا
    for pdf_path in data_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        raw_pages = loader.load()
        
        # هنلف على كل صفحة ونظبط الـ Metadata بتاعتها
        for page in raw_pages:
            # إضافة اسم الملف
            page.metadata["document_name"] = pdf_path.name
            
            # تظبيط رقم الصفحة عشان يبدأ من 1 بدل صفر
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1
            
            pages.append(page)
            
    return pages


def chunk_documents(pages):
    """
    Function to chunk pages smartly and add a unique chunk_id for citation.
    """
    # هنستخدم نفس الـ Section-aware splitter اللي جربناه
    aware_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,
        chunk_overlap=config.CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # تقطيع الصفحات
    chunks = aware_splitter.split_documents(pages)
    
    # إعطاء ID لكل Chunk عشان نقدر نتتبعه
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"
        
    return chunks
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def get_embedding_function():
    """
    Function to load the local embedding model (~100MB).
    """
    # هنستخدم FastEmbed عشان سريع ومناسب جداً للـ Local execution
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embeddings
from langchain_chroma import Chroma

def build_index(chunks):
    """
    Function to store embedded chunks in a local Chroma vector database.
    """
    embeddings = get_embedding_function()
    
    # بنعمل الـ Vector DB ونخزن الداتا في فولدر اسمه chroma_db عشان منضطرش نعملها Embed كل مرة
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectordb