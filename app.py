import os
import zipfile
import numpy as np
import tempfile
import logging
from flask import Flask, request, render_template

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image

# 🔧 Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Flask app
app = Flask(__name__)

# -----------------------------
# ✅ Class Names & Translations
# -----------------------------
twi_translations = {
    'Accidents and disaster': 'Asiane ne Amanehunu', 'Agriculture': 'Kuadwuma',
    'Architecture': 'Adan mu nhyehyɛe', 'Arts and crafts': 'Adwinneɛ ne nsaanodwuma',
    'Automobile': 'Kaa/ Kwan so nnwuma deɛ', 'Construction': 'Adesie',
    'Culture': 'Amammerɛ', 'Disabilities': 'Dɛmdi ahorow', 'Economy': 'Sikasɛm ne Ahonya ho nsɛm',
    'Education': 'Nwomasua/Adesua', 'Energy': 'Ahoɔden', 'Engineering': 'Mfiridwuma',
    'Entertainment': 'Anigyedeɛ', 'Ethnicity people and race': 'Mmusuakuw mu nnipa ne abusuakuw',
    'Family and society': 'Abusua ne Ɔmanfoɔ', 'Fashion and clothing': 'Ahosiesie ne Ntadeɛ',
    'Fauna and flora': 'Mmoa ne Nnua', 'Food and drink': 'Aduane ne Nsa', 'Funeral': 'Ayie',
    'Furniture': 'Efie adeɛ / Efie hyehyeɛ', 'Geography': 'Asase ho nimdeɛ',
    'Governance': 'Nniso nhyehyɛe', 'Health and medicine': 'Apɔmuden ne Nnuro',
    'History': 'Abakɔsɛm', 'Home and housing': 'Efie ne Tenabea', 'Hospitality': 'Ahɔhoyɛ',
    'Immigration': 'Atubrafo ho nsɛm', 'Justice and law enforcement': 'Atɛntenenee ne Mmara banbɔ',
    'Languages and Communication': 'Kasa ne Nkitahodie', 'Leisure': 'Ahomegyeɛ', 'Lifestyle': 'Abrateɛ',
    'Love and romance': 'Ɔdɔ ne Ɔdɔ ho nsɛm', 'Marine': 'Ɛpo mu nsɛm', 'Mining': 'Awuto fagude',
    'Movie cinema and theatre': 'Sinima ne Agorɔhwɛbea', 'Music and dance': 'Nnwom ne Asaw',
    'Nature': 'Abɔdeɛ', 'News': 'Kaseɛbɔ', 'Politics': 'Amammuisɛm', 'Religion': 'Gyidi ne Nsom',
    'Sanitation': 'Ahoteɛ', 'Science': 'Saense', 'Security': 'Banbɔ', 'Sports': 'Agodie',
    'Technology': 'Tɛknɔlɔgyi', 'Trading and commerce': 'Dwadie ne Nsesaguoɔ',
    'Transportation': 'Akwantuo', 'Travel and tourism': 'Akwantuɔ ne Ahɔhoɔ',
    'Weather and climate': 'Ewiem tebea ne Ewiem nhyehyɛeɛ'
}
class_names = list(twi_translations.keys())

# -----------------------------
# ✅ Model extraction + load
# -----------------------------
MODEL_ZIP_PATH = "fine_tuned_model_3.0.zip"
MODEL_EXTRACTED_PATH = "fine_tuned_model_3.0.keras"

def extract_model_from_zip():
    if not os.path.exists(MODEL_EXTRACTED_PATH):
        logger.info("📦 Extracting model from ZIP...")
        try:
            with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")
            logger.info("✅ Model extracted successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to extract model: {e}")
            return False
    else:
        logger.info("📁 Model already extracted.")
    return True

def load_model_safely():
    if not extract_model_from_zip():
        return None
    try:
        model = load_model(MODEL_EXTRACTED_PATH)
        logger.info("✅ Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}", exc_info=True)
        return None

# -----------------------------
# ✅ Image preprocessing
# -----------------------------
def process_uploaded_image(uploaded_file, img_size=(224, 224)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            class_dir = os.path.join(tmpdir, "class0")
            os.makedirs(class_dir, exist_ok=True)
            uploaded_file.seek(0)
            image_path = os.path.join(class_dir, uploaded_file.filename)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())
            dataset = image_dataset_from_directory(
                tmpdir, image_size=img_size, batch_size=1, shuffle=False
            )
            for images, _ in dataset.take(1):
                return images.numpy()
    except Exception as e:
        logger.warning(f"⚠️ Dataset method failed: {e}. Trying PIL fallback.")
        uploaded_file.seek(0)
        return process_uploaded_image_pil(uploaded_file, img_size)

def process_uploaded_image_pil(uploaded_file, img_size=(224, 224)):
    try:
        image = Image.open(uploaded_file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(img_size, Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"❌ PIL image processing failed: {e}")
        return None

# -----------------------------
# ✅ Model initialization
# -----------------------------
logger.info("🚀 Initializing Twi Image Classifier...")
model = load_model_safely()

# -----------------------------
# ✅ Flask routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if model is None:
        return render_template("index.html", predictions=None, warning="Model not loaded.")

    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("index.html", predictions=None, warning="Please upload an image.")

        image_file = request.files["image"]
        img_tensor = process_uploaded_image(image_file)

        if img_tensor is None:
            return render_template("index.html", predictions=None, warning="Failed to process image.")

        try:
            preds = model.predict(img_tensor, verbose=0)[0]
            top_indices = preds.argsort()[::-1][:3]
            top_preds = [(class_names[i], twi_translations.get(class_names[i], "❓"), float(preds[i])) for i in top_indices]

            if top_preds[0][2] >= 0.5:
                return render_template("index.html", predictions=top_preds, warning=None)
            else:
                return render_template("index.html", predictions=[], warning="Gyidie no nsɔ 50%, enti yɛrentumi nka")
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return render_template("index.html", predictions=None, warning="Prediction failed.")

    return render_template("index.html", predictions=None, warning=None)

@app.route("/health")
def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__,
        "classes_count": len(class_names)
    }

# -----------------------------
# ✅ Run the Flask app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
