from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import io
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import base64
import re

# ---------------------- FastAPI Setup ---------------------- #
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Load MobileNetV2 ---------------------- #
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# ---------------------- Firebase Setup ---------------------- #
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------------------- Helper Functions ---------------------- #
def get_image_features(img: Image.Image):
    img = img.resize((224, 224)).convert("RGB")
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_image_from_source(src: str):
    """Load image from URL or base64 string."""
    try:
        # Remove possible data URI prefix
        if src.startswith("data:image"):
            src = re.sub('^data:image/.+;base64,', '', src)

        if src.startswith("http"):
            response = requests.get(src, timeout=5)
            img = Image.open(io.BytesIO(response.content))
        else:
            img_data = base64.b64decode(src)
            img = Image.open(io.BytesIO(img_data))
        img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"Skipping image: {e}")
        return None

# ---------------------- API Endpoint ---------------------- #
@app.post("/find-similar-products")
async def find_similar_products(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return {"error": "Only image files are allowed"}

        contents = await file.read()
        uploaded_img = Image.open(io.BytesIO(contents)).convert("RGB")
        uploaded_features = get_image_features(uploaded_img)

        products_ref = db.collection("products")
        docs = products_ref.stream()

        similar_products = []

        for doc in docs:
            data = doc.to_dict()
            image_sources = data.get("images") or [data.get("imageUrl")]
            if not image_sources:
                continue

            product_max_similarity = 0

            for src in image_sources:
                img = load_image_from_source(src)
                if img is None:
                    continue

                features = get_image_features(img)
                similarity = cosine_similarity(uploaded_features, features)
                if similarity > product_max_similarity:
                    product_max_similarity = similarity

            if product_max_similarity > 0.7:
                similar_products.append({
                    "id": doc.id,
                    "title": data.get("title"),
                    "price": data.get("price"),
                    "images": image_sources,
                    "description": data.get("description"),
                    "similarity": float(product_max_similarity)
                })

        similar_products.sort(key=lambda x: x["similarity"], reverse=True)
        return {"results": similar_products}

    except Exception as e:
        return {"error": str(e)}
