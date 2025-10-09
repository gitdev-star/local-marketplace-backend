from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import io
from PIL import Image
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import base64
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ---------------------- FastAPI Setup ---------------------- #
app = FastAPI(title="Local Marketplace Image Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Load MobileNetV2 ---------------------- #
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# ThreadPoolExecutor for running blocking tasks (like model prediction)
executor = ThreadPoolExecutor(max_workers=4)

# ---------------------- Firebase Setup ---------------------- #
firebase_json = os.environ.get("FIREBASE_CREDENTIALS")

try:
    if firebase_json:
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
    elif os.path.exists("firebase_key.json"):
        cred = credentials.Certificate("firebase_key.json")
    else:
        raise RuntimeError(
            "Firebase credentials not found. Set FIREBASE_CREDENTIALS or place firebase_key.json in project folder."
        )

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

    db = firestore.client()
except Exception as e:
    raise RuntimeError(f"Failed to initialize Firebase: {e}")

# ---------------------- Helper Functions ---------------------- #
def extract_features(img: Image.Image) -> np.ndarray:
    """Blocking: Extract feature vector using MobileNetV2."""
    img = img.resize((224, 224)).convert("RGB")
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features[0]

async def get_image_features_async(img: Image.Image) -> np.ndarray:
    """Run blocking feature extraction in a separate thread."""
    loop = asyncio.get_running_loop()
    features = await loop.run_in_executor(executor, extract_features, img)
    return features

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_image_from_source(src: str) -> Image.Image | None:
    try:
        if src.startswith("data:image"):
            src = re.sub(r'^data:image/.+;base64,', '', src)
        if src.startswith("http"):
            response = requests.get(src, timeout=5)
            img = Image.open(io.BytesIO(response.content))
        else:
            img_data = base64.b64decode(src)
            img = Image.open(io.BytesIO(img_data))
        return img.convert("RGB")
    except Exception as e:
        print(f"Skipping image: {e}")
        return None

def get_base64_from_url(url: str) -> dict | None:
    """Fetch image from URL and convert to Base64 with MIME type."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "image/jpeg")
            b64_data = base64.b64encode(response.content).decode("utf-8")
            return {"mime": content_type, "data": b64_data}
        else:
            return None
    except Exception as e:
        print(f"Failed to fetch image: {e}")
        return None

# ---------------------- API Endpoint ---------------------- #
@app.post("/find-similar-products")
async def find_similar_products(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return {"error": "Only image files are allowed."}

        contents = await file.read()
        uploaded_img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run feature extraction asynchronously
        uploaded_features = await get_image_features_async(uploaded_img)

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

                features = await get_image_features_async(img)
                similarity = cosine_similarity(uploaded_features, features)
                if similarity > product_max_similarity:
                    product_max_similarity = similarity

            if product_max_similarity > 0.3:
                # Convert all images to Base64 with MIME type
                base64_images = []
                for src in image_sources:
                    if src.startswith("http"):  # Firebase URL
                        b64_dict = get_base64_from_url(src)
                        if b64_dict:
                            base64_images.append(b64_dict)
                    elif src.startswith("data:image"):  # Already a data URI
                        # Extract the MIME type and base64 data
                        match = re.match(r'data:([^;]+);base64,(.+)', src)
                        if match:
                            base64_images.append({"mime": match.group(1), "data": match.group(2)})
                        else:
                            base64_images.append({"mime": "image/jpeg", "data": src})
                    else:  # Assume it's raw base64
                        # Remove any data URI prefix if present
                        clean_b64 = re.sub(r'^data:image/.+;base64,', '', src)
                        base64_images.append({"mime": "image/jpeg", "data": clean_b64})

                similar_products.append({
                    "id": doc.id,
                    "title": data.get("title"),
                    "price": data.get("price"),
                    "images": base64_images,
                    "description": data.get("description"),
                    "similarity": float(product_max_similarity)
                })

        similar_products.sort(key=lambda x: x["similarity"], reverse=True)
        return {"results": similar_products}

    except Exception as e:
        return {"error": str(e)}
