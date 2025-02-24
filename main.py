import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import pandas as pd
import numpy as np
import re  # Untuk menghapus plt.show()
import json


# Load API Key dari .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API Key untuk Gemini tidak ditemukan! Pastikan GEMINI_API_KEY ada di .env")

genai.configure(api_key=api_key)

# Inisialisasi FastAPI
app = FastAPI()

# Konfigurasi CORS agar frontend dapat mengakses API
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
    
@app.get("/")
async def root():
    return {"message": "API is running!"}