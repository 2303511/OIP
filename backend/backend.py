import logging
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import requests
import re

# -------------------------------
# Setup Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# Load Model
# -------------------------------
try:
    model = tf.keras.models.load_model("agricultural_pest_model_phase1.h5")
    logger.info(f" Model architecture: {model.summary()}")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

class_names = [
    'ants',
    'bees',
    'beetle',
    'catterpillar',
    'earthworms',
    'earwig',
    'grasshopper',
    'moth',
    'slug',
    'snail',
    'wasp',
    'weevil'
]

# -------------------------------
# LLM Config
# -------------------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"

app = FastAPI()

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Helper: Preprocess Image
# -------------------------------
def preprocess_image(image_bytes):
    logger.info("📷 Preprocessing image for CNN input (224x224x3)...")

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Keep 3 channels
    img = img.resize((224, 224))                              # Match model input

    img_array = np.array(img) / 255.0                         # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)             # Add batch dimension → (1, 224, 224, 3)

    logger.info(f"✅ Preprocessed image shape: {img_array.shape}")
    return img_array

# -------------------------------
# Endpoint: /predict
# -------------------------------
@app.post("/predict")

async def predict(file: UploadFile = File(...), crop: str = Form(...)):
    try:
        logger.info(f"📥 Received file: {file.filename}, Crop: {crop}")

        # Read and preprocess image
        contents = await file.read()
        image = preprocess_image(contents)

        # Predict
        logger.info("🔍 Running prediction...")
        predictions = model.predict(image)
        print(predictions)  # Debugging output
        confidence = float(np.max(predictions))
        class_index = int(np.argmax(predictions))
        predicted_class = class_names[class_index]
        logger.info(f"✅ Predicted: {predicted_class} ({confidence:.2f})")

        # Crop-specific logic
        crop_context_status = f"Insect is {'harmful' if predicted_class != 'spider_mite' else 'beneficial'} to {crop}"
        # llm_response = f"The insect '{predicted_class}' is considered {'harmful' if predicted_class != 'spider_mite' else 'beneficial'} for {crop} crops. Please monitor accordingly."
        logger.info("🧠 Generated LLM response. \n =========================================")

        # ---- Construct LLM prompt here (AFTER prediction) ----
        prompt = f"""
        You are an expert in organic farming and pest management.
        An image of a {crop} crop was classified as: {predicted_class} (confidence {confidence:.1%}).
        
        Please provide:
        1. A short explanation of this pest and its impact on {crop}.
        2. Organic treatment steps (non-chemical).
        3. Prevention tips for farmers.
        
        Format:
        - Diagnosis:
        - Treatment:
        - Prevention:
        """

        # ---- Call LLM ----
        try:
            lm_payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "Answer concisely and avoid <think> tags."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
        
            lm_response = requests.post(LM_STUDIO_URL, json=lm_payload)
            lm_response.raise_for_status()
            lm_data = lm_response.json()
        
            # Extract response text
            content = lm_data["choices"][0]["message"]["content"]
        
            # Strip out <think> sections
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        
            llm_response = content
        
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            llm_response = "Could not generate LLM explanation."

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "crop_context_status": crop_context_status,
            "llm_response": llm_response
        }

    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting FastAPI server...")
    uvicorn.run("backend:app", port=8000, reload=True)
