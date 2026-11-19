from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import predict

app = FastAPI(title="LeishScan AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online", "service": "LeishScan AI Inference Engine"}

@app.post("/predict")
async def run_prediction(
    lesion_image: UploadFile = File(...),
    microscopy_image: UploadFile = File(...)
):
    try:
        lesion_bytes = await lesion_image.read()
        microscopy_bytes = await microscopy_image.read()
        result = predict(lesion_bytes, microscopy_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
