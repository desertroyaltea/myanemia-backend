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

# Define your model architectures (copy from your training code)
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        # TODO: Add your actual model architecture here
        # This is a placeholder - replace with your actual architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, 2)  # Binary classification
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # TODO: Add your actual model architecture here
        # This is a placeholder - replace with your actual architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Linear(128, 1)  # Single value output
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# Load models at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cls_model = ClassificationModel().to(device)
cls_model.load_state_dict(torch.load('models/best_cls_model.pth', map_location=device))
cls_model.eval()

hb_model = RegressionModel().to(device)
hb_model.load_state_dict(torch.load('models/best_hb_model.pth', map_location=device))
hb_model.eval()

# Image preprocessing
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
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def detect_eyelid_region(image: Image.Image) -> Image.Image:
    """
    Detect and crop the eyelid region from the image.
    This is a simplified version - adjust based on your actual preprocessing.
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    
    # Simple center crop as placeholder
    # TODO: Replace with your actual eyelid detection logic
    height, width = img_array.shape[:2]
    crop_size = min(height, width) // 2
    center_y, center_x = height // 2, width // 2
    
    y1 = max(0, center_y - crop_size // 2)
    y2 = min(height, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(width, center_x + crop_size // 2)
    
    cropped = img_array[y1:y2, x1:x2]
    return Image.fromarray(cropped)

@app.get("/")
async def root():
    return {"message": "MyAnemia API is running", "status": "healthy"}

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(request: AnalyzeRequest):
    try:
        # Decode the image
        image = decode_base64_image(request.image_b64)
        
        # Detect and crop eyelid region
        cropped_image = detect_eyelid_region(image)
        
        # Preprocess for model
        input_tensor = transform(cropped_image).unsqueeze(0).to(device)
        
        # Run classification model
        with torch.no_grad():
            cls_output = cls_model(input_tensor)
            is_anemic = torch.softmax(cls_output, dim=1)[0][1].item() > 0.5
        
        # Run regression model
        with torch.no_grad():
            hb_value = hb_model(input_tensor).item()
        
        # Ensure reasonable Hb range (4-18 g/dL)
        hb_value = max(4.0, min(18.0, hb_value))
        
        # Encode cropped image for response
        crop_b64 = encode_image_to_base64(cropped_image)
        
        return AnalyzeResponse(
            hb_value=hb_value,
            crop_b64=crop_b64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}