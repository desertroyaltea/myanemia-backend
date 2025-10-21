from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import cv2
import numpy as np

app = FastAPI()

# CORS middleware to allow your Netlify frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Netlify URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Your Exact Model Architectures from train_tuned_cnn.py ---
class TunedHbEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)

class TunedAnemiaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Load models at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the Hemoglobin regression model
hb_model = TunedHbEstimator().to(device)
hb_model.load_state_dict(torch.load('models/best_hb_model.pth', map_location=device))
hb_model.eval()
print("✓ Hb model loaded")

# Load the Anemia classification model
cls_model = TunedAnemiaClassifier().to(device)
cls_model.load_state_dict(torch.load('models/best_cls_model.pth', map_location=device))
cls_model.eval()
print("✓ Classification model loaded")

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class AnalyzeRequest(BaseModel):
    image_b64: str

class AnalyzeResponse(BaseModel):
    hb_value: float
    crop_b64: str

def decode_base64_image(image_b64: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    return image.convert('RGB')

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def detect_eyelid_region(image: Image.Image) -> Image.Image:
    """
    Detect and crop the eyelid region from the image.
    This is a simple center crop. If you have a specific eyelid detection
    algorithm from your preprocessing, replace this function.
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Simple center crop (adjust based on your preprocessing)
    crop_size = min(height, width)
    center_y, center_x = height // 2, width // 2
    
    y1 = max(0, center_y - crop_size // 2)
    y2 = min(height, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(width, center_x + crop_size // 2)
    
    cropped = img_array[y1:y2, x1:x2]
    
    # Resize to square if needed
    cropped_pil = Image.fromarray(cropped)
    cropped_pil = cropped_pil.resize((224, 224), Image.LANCZOS)
    
    return cropped_pil

@app.get("/")
async def root():
    return {
        "message": "MyAnemia API is running",
        "status": "healthy",
        "models": {
            "hemoglobin_estimator": "TunedHbEstimator",
            "anemia_classifier": "TunedAnemiaClassifier"
        }
    }

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(request: AnalyzeRequest):
    try:
        # Decode the image
        image = decode_base64_image(request.image_b64)
        
        # Detect and crop eyelid region
        cropped_image = detect_eyelid_region(image)
        
        # Preprocess for model
        input_tensor = transform(cropped_image).unsqueeze(0).to(device)
        
        # Run hemoglobin regression model
        with torch.no_grad():
            hb_value = hb_model(input_tensor).squeeze().item()
        
        # Run classification model (optional, for additional validation)
        with torch.no_grad():
            cls_output = cls_model(input_tensor)
            cls_probs = torch.softmax(cls_output, dim=1)
            is_anemic = cls_probs[0][1].item() > 0.5
        
        # Ensure reasonable Hb range (4-18 g/dL for safety)
        hb_value = max(4.0, min(18.0, hb_value))
        
        # Encode cropped image for response
        crop_b64 = encode_image_to_base64(cropped_image)
        
        return AnalyzeResponse(
            hb_value=round(hb_value, 2),
            crop_b64=crop_b64
        )
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "device": str(device)
    }