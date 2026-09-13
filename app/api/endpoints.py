from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.detector import run_detection
from app.services.reasoning import answer_question

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        detections = run_detection(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    return {"filename": file.filename, "count": len(detections), "detections": detections}


@router.post("/ask")
async def ask(file: UploadFile = File(...), question: str = Form(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(question, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

    return result