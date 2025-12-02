from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import pickle
import json
import pandas as pd
import uvicorn
from typing import Optional, List

# ================================
# 🔹 Carregar modelos e estruturas
# ================================
import os

# Caminho da raiz do projeto (um nível acima de backend/)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

# Modelo
model_path = os.path.join(ROOT_DIR, "modelo_dengue.h5")
model = tf.keras.models.load_model(model_path)

# Scaler
scaler_path = os.path.join(ROOT_DIR, "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Colunas
colunas_path = os.path.join(ROOT_DIR, "colunas.json")
with open(colunas_path, "r") as f:
    colunas = json.load(f)
# ================================
# 🔹 FastAPI
# ================================
app = FastAPI(
    title="Modelo de Previsão Dengue",
    description="API que prevê o tempo de encerramento com base nos dados processados",
    version="1.0.0"
)

# ================================
# 🔹 CORS
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 🔹 Schema da requisição
# ================================
class PredictRequest(BaseModel):
    tempo_sin_pri_notific: float
    tempo_invest_encerrar: float
    unidade: str

# ================================
# 🔹 Extrair unidades
# ================================
def extract_unidades_from_colunas(cols: List[str]) -> List[str]:
    unidades = []
    for c in cols:
        if isinstance(c, str) and c.startswith("UNIDADE_"):
            unidades.append(c.replace("UNIDADE_", ""))
    return unidades

UNIDADES = extract_unidades_from_colunas(colunas)

# ================================
# 🔹 Função de previsão
# ================================
def fazer_previsao(tempo_sin, tempo_invest, unidade):
    if tempo_sin is None or tempo_invest is None or not unidade:
        raise ValueError("Preencha todos os campos: tempo_sin_pri_notific, tempo_invest_encerrar e unidade")

    # Criar linha com todas as colunas
    linha = {col: 0 for col in colunas}
    linha["TEMPO_SIN_PRI_NOTIFIC"] = float(tempo_sin)
    linha["TEMPO_INVEST_ENCERRA"] = float(tempo_invest)

    # Dummy da unidade
    unidade_col = f"UNIDADE_{unidade}"
    if unidade_col in linha:
        linha[unidade_col] = 1

    df = pd.DataFrame([linha])

    # Escalar
    try:
        X_scaled = scaler.transform(df[colunas])
    except Exception as e:
        raise ValueError(f"Erro ao transformar features: {e}")

    # Prever
    pred = model.predict(X_scaled)[0][0]

    # Garantir que não seja negativo
    pred = max(pred, 0)

    return float(pred)

# ================================
# 🔹 Endpoints
# ================================
@app.get("/unidades")
def get_unidades():
    return UNIDADES

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        resultado = fazer_previsao(
            tempo_sin=data.tempo_sin_pri_notific,
            tempo_invest=data.tempo_invest_encerrar,
            unidade=data.unidade
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"previsao": resultado}

# ================================
# 🔹 Rodar local
# ================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
