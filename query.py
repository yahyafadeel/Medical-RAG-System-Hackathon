from langchain_chroma import Chroma
from ingest import get_embedding_function

def load_index():
    """
    Function to load the existing vector database from disk.
    """
    embeddings = get_embedding_function()
    # بنحمل الـ ChromaDB من الفولدر اللي حفظناها فيه امبارح
    vectordb = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    return vectordb

def retrieve(vectordb, query, k=3):
    """
    Function to search the vector database for the top-k most relevant chunks.
    """
    # بنبحث عن أقرب Chunks للسؤال ونرجعها مع الـ Score بتاع كل واحدة
    results = vectordb.similarity_search_with_relevance_scores(query, k=k)
    return results