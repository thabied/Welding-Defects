from fastapi import FastAPI, HTTPException, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI()

# Load the model
model = YOLO('/app/best.pt')

def preprocess_image(image_data, target_size=(160, 160)):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Resize the image
    img = cv2.resize(img, target_size)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        
        # Preprocess the image
        img = preprocess_image(contents)

        # Run inference
        results = model(img)

        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                predictions.append({
                    "class": class_name,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Welding Defect Detection API"}