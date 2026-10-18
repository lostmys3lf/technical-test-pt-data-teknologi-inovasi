import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# load semua artefak sekali waktu API nyala, bukan tiap request
with open(ARTIFACT_DIR / "ensemble_config.json") as f:
    config = json.load(f)

preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")

model_a = XGBClassifier()
model_a.load_model(ARTIFACT_DIR / "model_a_xgb.json")

session_b = ort.InferenceSession(str(ARTIFACT_DIR / "model_b_ann.onnx"))
input_name_b = session_b.get_inputs()[0].name

app = FastAPI(title="Sepsis Risk Prediction API")


class PatientData(BaseModel):
    heart_rate: float
    respiratory_rate: float
    temperature: float
    wbc_count: float
    lactate_level: Optional[float] = (
        None  # boleh kosong, nanti diisi median oleh imputer
    )
    age: float
    num_comorbidities: float


@app.post("/predict")
def predict(patient: PatientData):
    data = patient.model_dump()

    # susun input sesuai urutan fitur waktu training, None jadi NaN biar diisi imputer
    row = [[np.nan if data[f] is None else data[f] for f in config["features"]]]
    X = np.array(row, dtype=float)

    X_prep = preprocessor.transform(X)

    proba_a = model_a.predict_proba(X_prep)[:, 1][0]
    proba_b = session_b.run(None, {input_name_b: X_prep.astype(np.float32)})[0].ravel()[
        0
    ]

    # soft voting ensemble
    proba_ens = config["w_a"] * proba_a + config["w_b"] * proba_b
    prediction = int(proba_ens >= config["threshold"])

    return {
        "sepsis_risk_prediction": prediction,
        "risk_probability": round(float(proba_ens), 4),
    }
