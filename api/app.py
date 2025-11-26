from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.model import DynamicPricingModel

class PredictRequest(BaseModel):
    vehicleType: str
    timeIn: str  # format: 'DD-MM-YYYY HH:MM'
    timeOut: str
    paidAmt: float

app = FastAPI(title="Dynamic Pricing API")

# Allow CORS for local testing (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = DynamicPricingModel()
if not model.load():
    # If the model files are missing, the API will still start but return errors on predict.
    # Prefer running `python model/model.py` once to create artifacts.
    print("Warning: model artifacts not found. Run 'python model/model.py' to train and save the model.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model.is_trained}


@app.get("/")
def root():
    """Root endpoint - provides basic info and link to docs."""
    return JSONResponse({
        "message": "Dynamic Pricing API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "model_loaded": model.is_trained,
    })


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = model.predict(
            vehicle_type=req.vehicleType,
            time_in=req.timeIn,
            time_out=req.timeOut,
            paid_amt=req.paidAmt,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# If you want to run this module directly for development:
# uvicorn api.app:app --reload --port 8000
