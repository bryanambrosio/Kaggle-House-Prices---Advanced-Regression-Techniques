import pickle
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

# Inicializa a API
app = FastAPI()

# Caminho do modelo salvo
model_path = "models/house_pricing_model.pkl"

# Carregar o modelo treinado
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Definir a estrutura dos dados de entrada usando Pydantic
class HouseData(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    # Adicione todas as features que o modelo espera como entrada

@app.get("/")
def read_root():
    return {"message": "API de previsão de preços de casas está funcionando!"}

@app.post("/predict")
def predict(data: HouseData):
    # Converter os dados de entrada em DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Fazer a previsão
    prediction = model.predict(input_data)

    return {"prediction": prediction.tolist()}
