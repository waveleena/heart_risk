from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
predictor = HeartAttackPredictor('heart_risk/catboost_model.pkl')

class CSVPath(BaseModel):
    csv_path: str

@app.post("/predict")
def predict_from_csv(data: CSVPath):
    try:
        df = pd.read_csv(data.csv_path)
    except Exception as e:
        return {"error": str(e)}

    results = predictor.predict(df)
    return results
