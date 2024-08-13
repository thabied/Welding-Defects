# Welding Defects API

A YoloV8 Object Detction model for welding categorization, deployed as a local dockerized API

<img width="955" alt="Screenshot 2024-08-13 at 18 48 18" src="https://github.com/user-attachments/assets/2e129b57-13e5-4de4-8d63-3d983905635a">


- Trained on Dataset: https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection
- Using YoloV8n model from Ultralytics: https://docs.ultralytics.com
- Deployed using FastAPI as a Docker container

## Setup

1) Pull Docker image using `docker push thabiedmleng/welding-defect-api`
2) Run container using your container name and port of choice:
   `docker run -p 8000:8000 welding-defect-api:latest`
3) Query /predict endpoint using:
   `curl -X POST "http://localhost:8000/predict" -H "Content-Type: multipart/form-data" -F "file=@/path/to/images/test_image.jpg"`

   replacing `/path/to/images/test_image.jpg` with the absolute file path to your image

## Output

Output format is a dictionary with the following format:

`{"predictions":[{"class": ,"confidence": ,"bbox": },{"class": ,"confidence": ,"bbox": }]}`

where 
- "class": has 3 labels ("Good Weld","Bad Weld","Defect")
- "confidence": is a probabilistic score between 0 and 1
- "bbox": are pixel coordinates for bounding boxes 
