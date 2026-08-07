from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib  # <--- Important: joblib import kar liya hai

# Global variable for Pipeline
pipeline = None

# 1. Modern Lifespan Event (Server start/shutdown handling)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        pipeline = joblib.load('xgb_churn_pipeline.pkl')
        print("✅ XGBoost Pipeline loaded successfully into FastAPI!")
    except Exception as e:
        print(f"❌ Failed to load model pipeline: {str(e)}")
    yield
    # Clear memory on server shutdown
    pipeline = None

# 2. FastAPI Instance Initialize
app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI Service serving XGBoost Pipeline Model",
    version="1.0",
    lifespan=lifespan  # <--- Clean & modern startup handler
)

# 3. Pydantic Schema (Data Types for Input Validation)
class CustomerInput(BaseModel):
    Age: int
    Tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Gender: str
    Contract: str
    PaymentMethod: str

# 4. Root Health-Check Endpoint
@app.get("/")
def home():
    return {"status": "Online", "message": "FastAPI Churn Inference Engine Active"}

# 5. Prediction API Endpoint
@app.post("/predict")
def predict_churn(customer: CustomerInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded.")
    
    # JSON request ko Pandas DataFrame mein convert kar rahe hain
    raw_data = pd.DataFrame([customer.dict()])
    
    # Model pipeline automatic preprocessing aur prediction karegi
    churn_proba = float(pipeline.predict_proba(raw_data)[0][1])
    is_churn = int(churn_proba > 0.5)
    
    return {
        "status": "success",
        "churn_prediction": is_churn,
        "churn_risk_label": "High Risk" if is_churn == 1 else "Low Risk",
        "churn_probability": round(churn_proba * 100, 2),
        "recommendation": "High churn risk! Consider offering a discounted long-term contract." if is_churn == 1 else "Customer is stable."
    }