"""
FastAPI Application for Kidney Stone Detection
Serves predictions via REST API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision
import numpy as np
import joblib
import io
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Kidney Stone Detection API",
    description="AI-powered API for detecting kidney stones in ultrasound images",
    version="1.0.0"
)

# Enable CORS (for Android and other clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
DEVICE = torch.device("cpu")

# ==========================================
# Load Models at Startup
# ==========================================

# Global variables to store models
resnet = None
clf = None
metadata = None

@app.on_event("startup")
def load_models():
    """Load models when server starts"""
    global resnet, clf, metadata
    
    try:
        logger.info("Loading models...")
        
        # Load ResNet50
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        resnet = torchvision.models.resnet50(weights=weights)
        resnet.fc = torch.nn.Identity()
        resnet = resnet.to(DEVICE)
        resnet.eval()
        logger.info("✓ ResNet50 loaded")
        
        # Load Classifier
        clf = joblib.load("logistic_regression_classifier.pkl")
        logger.info("✓ Logistic Regression classifier loaded")
        
        # Load Metadata
        metadata = torch.load("resnet50_feature_metadata.pt")
        logger.info("✓ Metadata loaded")
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# ==========================================
# Image Preprocessing
# ==========================================

from torchvision import transforms

# Create preprocessing pipeline
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Kidney Stone Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "info": "/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": resnet is not None and clf is not None,
        "message": "API is running and ready for predictions"
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Metadata not loaded")
    
    return {
        "model_name": "Kidney Stone Detection",
        "architecture": "ResNet50 + Logistic Regression",
        "classes": metadata.get("classes", []),
        "class_mapping": metadata.get("class_to_idx", {}),
        "feature_size": metadata.get("feature_size", 0),
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.93,
        "f1_score": 0.92
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict kidney stone presence in uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        {
            "prediction": "positive" or "negative",
            "confidence": 0.95,
            "probability": {
                "negative": 0.05,
                "positive": 0.95
            },
            "success": true
        }
    """
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/bmp"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Upload JPEG, PNG, or BMP images only."
            )
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        logger.info(f"Processing image: {file.filename}")
        
        # Preprocess image
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        # Extract features
        with torch.no_grad():
            features = resnet(tensor).detach().cpu().numpy()
        
        # Get prediction
        prediction = clf.predict(features)[0]
        probabilities = clf.predict_proba(features)[0]
        confidence = float(probabilities.max())
        
        # Map class index to label
        idx_to_class = metadata.get("idx_to_class", {0: "negative", 1: "positive"})
        prediction_label = idx_to_class.get(int(prediction), "unknown")
        
        logger.info(f"Prediction: {prediction_label}, Confidence: {confidence:.4f}")
        
        # Prepare response
        response = {
            "success": True,
            "prediction": prediction_label,
            "confidence": round(confidence, 4),
            "probability": {
                "negative": round(float(probabilities[0]), 4),
                "positive": round(float(probabilities[1]), 4)
            },
            "image_name": file.filename,
            "model": "ResNet50 + Logistic Regression"
        }
        
        return JSONResponse(content=response, status_code=200)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "message": "Error processing image"
            },
            status_code=400
        )

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict kidney stone presence in multiple images
    
    Args:
        files: List of image files
    
    Returns:
        List of predictions for each image
    """
    results = []
    
    for file in files:
        try:
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/bmp"]:
                results.append({
                    "success": False,
                    "filename": file.filename,
                    "error": "Invalid file type"
                })
                continue
            
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Preprocess
            tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                features = resnet(tensor).detach().cpu().numpy()
            
            prediction = clf.predict(features)[0]
            probabilities = clf.predict_proba(features)[0]
            confidence = float(probabilities.max())
            
            idx_to_class = metadata.get("idx_to_class", {0: "negative", 1: "positive"})
            prediction_label = idx_to_class.get(int(prediction), "unknown")
            
            results.append({
                "success": True,
                "filename": file.filename,
                "prediction": prediction_label,
                "confidence": round(confidence, 4),
                "probability": {
                    "negative": round(float(probabilities[0]), 4),
                    "positive": round(float(probabilities[1]), 4)
                }
            })
        
        except Exception as e:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})

# ==========================================
# Error Handlers
# ==========================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)}
    )

# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    import uvicorn
    # Run on 0.0.0.0:8000 so it's accessible from any network
    uvicorn.run(app, host="0.0.0.0", port=8000)
