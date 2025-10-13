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
from typing import List, Dict, Optional

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

# ThreadPoolExecutor for running blocking tasks
executor = ThreadPoolExecutor(max_workers=4)

# ---------------------- Firebase Setup ---------------------- #
firebase_json = os.environ.get("FIREBASE_CREDENTIALS")

try:
    if firebase_json:
        cred_dict = json.loads(firebase_json)
        if 'private_key' in cred_dict:
            cred_dict['private_key'] = cred_dict['private_key'].replace('\\n', '\n')
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

# ---------------------- GitHub Configuration ---------------------- #
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USER = "gitdev-star"
GITHUB_REPO = "local-marketplace-images"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents"

# Cache for GitHub images to avoid repeated API calls
github_images_cache = {}

# ---------------------- Helper Functions ---------------------- #
def extract_features(img: Image.Image) -> np.ndarray:
    """Extract feature vector using MobileNetV2."""
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

def load_image_from_source(src: str) -> Optional[Image.Image]:
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

def get_base64_from_url(url: str) -> Optional[Dict]:
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

def normalize_folder_name(name: str) -> str:
    """Normalize folder/product names for matching."""
    return name.lower().strip().replace(" ", "").replace("-", "").replace("_", "")

def fetch_github_images_by_folder(path="models") -> Dict[str, List[str]]:
    """
    Fetch images from GitHub organized by folder name.
    Returns: {folder_name: [image_url1, image_url2, ...]}
    """
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    url = f"{GITHUB_API_URL}/{path}"
    
    folder_images = {}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            items = response.json()
            
            for item in items:
                if item["type"] == "dir":
                    folder_name = item["name"]
                    folder_images[folder_name] = []
                    
                    # Get images inside this folder
                    folder_response = requests.get(item["url"], headers=headers, timeout=10)
                    if folder_response.status_code == 200:
                        folder_contents = folder_response.json()
                        
                        for file in folder_contents:
                            if file["type"] == "file" and file["name"].lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                                folder_images[folder_name].append(file["download_url"])
            
            return folder_images
        else:
            print(f"GitHub API error: {response.status_code}")
            return {}
    except Exception as e:
        print(f"Error fetching GitHub images: {e}")
        return {}

def match_github_images_to_product(product_title: str, github_folders: Dict[str, List[str]]) -> List[str]:
    """
    Match product title to GitHub folder and return image URLs.
    """
    normalized_title = normalize_folder_name(product_title)
    
    # Try exact match first
    for folder_name, images in github_folders.items():
        if normalize_folder_name(folder_name) == normalized_title:
            return images
    
    # Try partial match (folder name contains product title or vice versa)
    for folder_name, images in github_folders.items():
        normalized_folder = normalize_folder_name(folder_name)
        if normalized_title in normalized_folder or normalized_folder in normalized_title:
            return images
    
    return []

# ---------------------- Main API Endpoint ---------------------- #
@app.post("/find-similar-products")
async def find_similar_products(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return {"error": "Only image files are allowed."}

        contents = await file.read()
        uploaded_img = Image.open(io.BytesIO(contents)).convert("RGB")
        uploaded_features = await get_image_features_async(uploaded_img)

        # Fetch GitHub images organized by folder
        print("Fetching GitHub images...")
        github_folders = fetch_github_images_by_folder(path="models")
        print(f"Found {len(github_folders)} folders in GitHub")

        # Fetch Firestore products
        products_ref = db.collection("products")
        docs = products_ref.stream()

        similar_products = []

        for doc in docs:
            data = doc.to_dict()
            product_title = data.get("title", "")
            
            # Get images from Firestore
            firestore_images = data.get("images") or [data.get("imageUrl")]
            
            # Get matching images from GitHub based on product title
            github_images = match_github_images_to_product(product_title, github_folders)
            
            # Combine both sources (prioritize GitHub if available)
            all_image_sources = github_images + (firestore_images if firestore_images[0] else [])
            
            if not all_image_sources:
                continue

            product_max_similarity = 0

            # Calculate similarity using all available images
            for src in all_image_sources:
                img = load_image_from_source(src)
                if img is None:
                    continue

                features = await get_image_features_async(img)
                similarity = cosine_similarity(uploaded_features, features)
                if similarity > product_max_similarity:
                    product_max_similarity = similarity

            if product_max_similarity > 0.3:
                # Convert images to base64 for response
                base64_images = []
                
                # Process GitHub images first (higher priority)
                for url in github_images:
                    b64_dict = get_base64_from_url(url)
                    if b64_dict:
                        base64_images.append(b64_dict)
                
                # Add Firestore images if no GitHub images found
                if not base64_images:
                    for src in firestore_images:
                        if not src:
                            continue
                        if src.startswith("http"):
                            b64_dict = get_base64_from_url(src)
                            if b64_dict:
                                base64_images.append(b64_dict)
                        elif src.startswith("data:image"):
                            match = re.match(r'data:([^;]+);base64,(.+)', src)
                            if match:
                                base64_images.append({"mime": match.group(1), "data": match.group(2)})
                        else:
                            clean_b64 = re.sub(r'^data:image/.+;base64,', '', src)
                            base64_images.append({"mime": "image/jpeg", "data": clean_b64})

                similar_products.append({
                    "id": doc.id,
                    "title": product_title,
                    "price": data.get("price"),
                    "images": base64_images,
                    "description": data.get("description"),
                    "contacts": data.get("contacts", []),
                    "similarity": float(product_max_similarity),
                    "source": "github" if github_images else "firestore"
                })

        similar_products.sort(key=lambda x: x["similarity"], reverse=True)
        return {
            "results": similar_products,
            "total": len(similar_products),
            "github_folders_found": len(github_folders)
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# ---------------------- Additional Endpoints ---------------------- #
@app.get("/github-folders")
async def get_github_folders():
    """Get list of all folders in GitHub models directory."""
    folders = fetch_github_images_by_folder(path="models")
    return {
        "count": len(folders),
        "folders": {name: len(urls) for name, urls in folders.items()}
    }

@app.get("/test-match/{product_title}")
async def test_product_match(product_title: str):
    """Test endpoint to see which GitHub folder matches a product title."""
    github_folders = fetch_github_images_by_folder(path="models")
    matched_images = match_github_images_to_product(product_title, github_folders)
    
    return {
        "product_title": product_title,
        "normalized": normalize_folder_name(product_title),
        "matched_folder": next((name for name, imgs in github_folders.items() 
                                if normalize_folder_name(name) == normalize_folder_name(product_title)), None),
        "image_count": len(matched_images),
        "images": matched_images
    }

@app.get("/")
async def root():
    return {
        "message": "Local Marketplace Image Search API",
        "endpoints": {
            "POST /find-similar-products": "Upload image to find similar products",
            "GET /github-folders": "List all GitHub folders",
            "GET /test-match/{product_title}": "Test GitHub folder matching",
            "GET /test-firestore": "Test Firestore data structure"
        }
    }

@app.get("/test-firestore")
async def test_firestore():
    """Test endpoint to check Firestore product structure."""
    try:
        products_ref = db.collection("products")
        docs = products_ref.limit(3).stream()
        
        products = []
        for doc in docs:
            data = doc.to_dict()
            products.append({
                "id": doc.id,
                "title": data.get("title"),
                "has_images": bool(data.get("images")),
                "has_description": bool(data.get("description")),
                "has_contacts": bool(data.get("contacts")),
                "contacts": data.get("contacts", []),
                "all_fields": list(data.keys())
            })
        
        return {
            "total_checked": len(products),
            "products": products
        }
    except Exception as e:
        return {"error": str(e)}