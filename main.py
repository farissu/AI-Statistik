import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import psycopg2
import shutil
import pandas as pd
import numpy as np
import re  # Untuk menghapus plt.show()
import json

# LangChain & Vector Store
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load API Key dari .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API Key untuk Gemini tidak ditemukan! Pastikan GEMINI_API_KEY ada di .env")

genai.configure(api_key=api_key)

# Inisialisasi FastAPI
app = FastAPI()

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Logging untuk debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model request
class RAGRequest(BaseModel):
    question: str

# Model request untuk eksekusi kode Python
class CodeExecutionRequest(BaseModel):
    code: str

class PromptRequest(BaseModel):
    prompt: str

# Fungsi membersihkan kode dari AI
def clean_code(code: str) -> str:
    return code.replace("```python", "").replace("```", "").strip()
def clean_text(text: str) -> str:
    return text.replace("*", "").strip()

# Fungsi menghapus plt.show() sebelum dieksekusi
def remove_plt_show(code: str) -> str:
    return re.sub(r"\bplt\.show\(\)", "", code)

# Fungsi eksekusi kode dengan keamanan tambahan
def restricted_exec(code: str):
    safe_builtins = {
        "print": print,
        "len": len,
        "sum": sum,
        "range": range,
        "__import__": __import__  # Izinkan __import__ agar numpy & pandas bisa bekerja
    }

    safe_globals = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np
    }

    exec_locals = {}

    try:
        exec(code, safe_globals, exec_locals)
    except Exception as e:
        raise RuntimeError(f"Eksekusi gagal: {str(e)}")

    return exec_locals

# Endpoint untuk berinteraksi dengan Gemini AI
@app.post("/ask")
async def ask_gemini(request: PromptRequest):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(request.prompt)

        if hasattr(response, "text"):
            response_text = clean_code(response.text.strip())
            return JSONResponse(content={"script": response_text})

        return JSONResponse(content={"error": "❌ Tidak ada jawaban dari model."}, status_code=500)

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        return JSONResponse(content={"error": f"❌ Terjadi kesalahan: {str(e)}"}, status_code=500)

# Endpoint untuk mengeksekusi kode Python dan mengembalikan data dalam format Chart.js
@app.post("/execute")
async def execute_code(request: CodeExecutionRequest):
    try:
        code = remove_plt_show(clean_code(request.code))

        # Dictionary kosong untuk menyimpan hasil eksekusi
        exec_globals = {}

        # Eksekusi kode Python yang diberikan pengguna
        exec(code, {}, exec_globals)

        # Cari variabel yang berisi data
        labels = None
        data = None

        for var_name, var_value in exec_globals.items():
            if isinstance(var_value, list) and all(isinstance(i, str) for i in var_value):  
                labels = var_value  # Jika list berisi string, kemungkinan ini labels
            elif isinstance(var_value, list) and all(isinstance(i, (int, float)) for i in var_value):  
                data = var_value  # Jika list berisi angka, kemungkinan ini data angka
            elif isinstance(var_value, dict):  # Mendeteksi dictionary
                labels = list(var_value.keys())
                data = list(var_value.values())

        if labels and data:
            chart_data = {"labels": labels, "data": data}
            return JSONResponse(content={"chart_data": chart_data})
        
        return JSONResponse(content={"error": "❌ Tidak ditemukan data yang bisa digunakan."}, status_code=500)

    except Exception as e:
        logger.error(f"Error in /execute: {str(e)}")
        return JSONResponse(content={"error": f"❌ Terjadi kesalahan: {str(e)}"}, status_code=500)
    

# Endpoint untuk analisis teks menggunakan Gemini AI
@app.post("/analyze")
async def analyze_text(request: PromptRequest):
    try:
        model = genai.GenerativeModel("gemini-pro")
        custom_prompt = f"Kamu adalah seorang analis data profesional. Tolong analisa data berikut dan berikan reasoning yang mendalam: {request.prompt}"
        response = model.generate_content(custom_prompt)

        if hasattr(response, "text"):
            response_text = clean_text(clean_code(response.text.strip()))
            return JSONResponse(content={"analysis": response_text})

        return JSONResponse(content={"error": "❌ Tidak ada jawaban dari model."}, status_code=500)

    except Exception as e:
        logger.error(f"Error in /analyze: {str(e)}")
        return JSONResponse(content={"error": f"❌ Terjadi kesalahan: {str(e)}"}, status_code=500)
# Fungsi untuk mengambil data dari database
def fetch_data_from_db():
    """Mengambil data terbaru dari database dengan error handling"""
    try:
        with psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=int(os.getenv("DB_PORT", "5432"))
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, username, email FROM akun")  
                rows = cursor.fetchall()

        logger.info(f"✅ Database fetched rows: {rows}")  
        return [f"User ID: {row[0]}, Username: {row[1]}, Email: {row[2]}" for row in rows]

    except psycopg2.Error as e:
        logger.error(f"❌ Database error: {str(e)}")
        return []

# ---- 📌 Setup ChromaDB untuk RAG ----

# Hapus index lama untuk memastikan data terbaru
if os.path.exists("./chroma_db/index"):
    shutil.rmtree("./chroma_db/index")

# Ambil data terbaru dari database
documents = fetch_data_from_db()

# Cek apakah ada data dari database
if not documents:
    raise ValueError("❌ Tidak ada data dari database untuk dimasukkan ke ChromaDB!")

# Split dokumen sebelum masuk ke ChromaDB
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents(documents)

# Gunakan GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Simpan embedding ke ChromaDB
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Ambil 1 dokumen terkait

# Gunakan Gemini AI sebagai LLM
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"verbose": True}
)

# ---- 📌 Endpoint: Retrieval-Augmented Generation (RAG) ----
@app.post("/rag")
async def rag_query(request: RAGRequest):
    try:
        # Ambil dokumen terkait dari ChromaDB
        retrieved_docs = retriever.get_relevant_documents(request.question)
        logger.info(f"🔍 Retrieved {len(retrieved_docs)} documents from ChromaDB")

        # Jika tidak ada dokumen yang ditemukan
        if not retrieved_docs:
            return JSONResponse(content={"answer": "⚠️ No relevant documents found."})

        # Gunakan Gemini AI untuk menjawab dengan RetrievalQA
        answer = qa_chain.invoke({"query": request.question})

        logger.info(f"📝 Generated Answer: {answer}")

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        logger.error(f"❌ Error in RAG query: {str(e)}")
        return JSONResponse(content={"error": f"Terjadi kesalahan: {str(e)}"}, status_code=500)

@app.get("/")
async def root():
    return {"message": "API is running!"}