from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import predict

app = FastAPI(title="Insurance Fraud Detection App")

class ApplicationData(BaseModel):
    Customer_Age: int
    Claim_Amount: float
    Claim_History: int
    Policy_Type: str
    Incident_Severity: str
    Claim_Frequency: int
    Claim_Description: str
    Gender: str
    Marital_Status: str
    Occupation: str
    Income_Level: int
    Education_Level: str
    Location: str
    Behavioral_Data: str
    Purchase_History: str
    Policy_Start_Date: str
    Policy_Renewal_Date: str
    Interactions_with_Customer_Service: str
    Insurance_Products_Owned: str
    Coverage_Amount: float
    Premium_Amount: float
    Deductible: float
    Policy_Type_Dup: str
    Customer_Preferences: str
    Preferred_Communication_Channel: str
    Driving_Record: str
    Life_Events: str

@app.post("/predict")
def predict_fraud(data: ApplicationData):
    try:
        result = predict(data.dict())
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
