from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import cv2
import numpy as np
import base64

# === Init app ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === Class labels ===
class_names = [
    'brown_spot', 'corn_rust', 'corn_smut',
    'downy_mildew', 'grey_leaf_spot', 'healthy', 'leaf_blight'
]

# === Disease infos ===
class_infos = {
    "brown_spot": "Agent : *Physoderma maydis*. Taches jaunes rondes devenant brunes, affaiblissement des tiges.",
    "corn_rust": "Agent : *Puccinia sorghi*. Pustules poudreuses, peu de perte de rendement.",
    "corn_smut": "Agent : *Ustilago maydis*. Galles noires, réduction potentielle du rendement.",
    "downy_mildew": "Oomycètes comme *Peronosclerospora*. Feuilles pâles, croissance ralentie.",
    "grey_leaf_spot": "Agent : *Cercospora zeae-maydis*. Taches rectangulaires, perte de photosynthèse.",
    "healthy": "Aucune lésion. Photosynthèse optimale, rendement maximal.",
    "leaf_blight": "Agents : *Colletotrichum graminicola*, *Exserohilum turcicum*. Taches ovales, sénescence accélérée."
}

# === Device & model ===
from efficientnet_pytorch import EfficientNet

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_name('efficientnet-b4',num_classes=7)
state_dict = torch.load("damage_analysis_best_model.pt", map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Prediction helper ===
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
        return class_names[pred_idx], conf.item(), probs.cpu().numpy()

# === Webcam capture ===
def capture_webcam_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    # Convert to RGB and PIL format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    # Encode image to base64 for HTML display
    _, buffer = cv2.imencode('.jpg', frame)
    b64_image = base64.b64encode(buffer).decode('utf-8')
    return pil_image, b64_image

# === Routes ===
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        class_name, confidence, _ = predict_image(image)
        results.append({
            "filename": file.filename,
            "predicted_class": class_name,
            "confidence": f"{confidence * 100:.2f}%",
            "explanation": class_infos[class_name]
        })
    return templates.TemplateResponse("result.html", {"request": request, "results": results})

@app.post("/predict_webcam", response_class=HTMLResponse)
async def predict_webcam(request: Request):
    image, b64_image = capture_webcam_image()
    if image is None:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Camera error."})
    class_name, confidence, _ = predict_image(image)
    result = {
        "filename": "webcam_capture.jpg",
        "predicted_class": class_name,
        "confidence": f"{confidence * 100:.2f}%",
        "explanation": class_infos[class_name],
        "img_data": b64_image
    }
    return templates.TemplateResponse("result_webcam.html", {"request": request, "result": result})
