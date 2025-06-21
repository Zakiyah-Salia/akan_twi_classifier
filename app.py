import os
import gdown
import tempfile
import shutil
import numpy as np
from flask import Flask, request, render_template

# NO MORE PATCHES. The environment is now correct.
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Twi class names
twi_translations = {
    'Accidents and disaster': 'Asiane ne Amanehunu', 'Agriculture': 'Kuadwuma', 'Architecture': 'Adan mu nhyehyɛe',
    'Arts and crafts': 'Adwinneɛ ne nsaanodwuma', 'Automobile': 'Kaa/ Kwan so nnwuma deɛ', 'Construction': 'Adesie',
    'Culture': 'Amammerɛ', 'Disabilities': 'Dɛmdi ahorow', 'Economy': 'Sikasɛm ne Ahonya ho nsɛm',
    'Education': 'Nwomasua/Adesua', 'Energy': 'Ahoɔden', 'Engineering': 'Mfiridwuma',
    'Entertainment': 'Anigyedeɛ', 'Ethnicity people and race': 'Mmusuakuw mu nnipa ne abusuakuw',
    'Family and society': 'Abusua ne Ɔmanfoɔ', 'Fashion and clothing': 'Ahosiesie ne Ntadeɛ',
    'Fauna and flora': 'Mmoa ne Nnua', 'Food and drink': 'Aduane ne Nsa', 'Funeral': 'Ayie',
    'Furniture': 'Efie adeɛ / Efie hyehyeɛ', 'Geography': 'Asase ho nimdeɛ', 'Governance': 'Nniso nhyehyɛe',
    'Health and medicine': 'Apɔmuden ne Nnuro', 'History': 'Abakɔsɛm', 'Home and housing': 'Efie ne Tenabea',
    'Hospitality': 'Ahɔhoyɛ', 'Immigration': 'Atubrafo ho nsɛm', 'Justice and law enforcement': 'Atɛntenenee ne Mmara banbɔ',
    'Languages and Communication': 'Kasa ne Nkitahodie', 'Leisure': 'Ahomegyeɛ', 'Lifestyle': 'Abrateɛ',
    'Love and romance': 'Ɔdɔ ne Ɔdɔ ho nsɛm', 'Marine': 'Ɛpo mu nsɛm', 'Mining': 'Awuto fagude',
    'Movie cinema and theatre': 'Sinima ne Agorɔhwɛbea', 'Music and dance': 'Nnwom ne Asaw', 'Nature': 'Abɔdeɛ',
    'News': 'Kaseɛbɔ', 'Politics': 'Amammuisɛm', 'Religion': 'Gyidi ne Nsom', 'Sanitation': 'Ahoteɛ',
    'Science': 'Saense', 'Security': 'Banbɔ', 'Sports': 'Agodie', 'Technology': 'Tɛknɔlɔgyi',
    'Trading and commerce': 'Dwadie ne Nsesaguoɔ', 'Transportation': 'Akwantuo', 'Travel and tourism': 'Akwantuɔ ne Ahɔhoɔ',
    'Weather and climate': 'Ewiem tebea ne Ewiem nhyehyɛeɛ'
}
class_names = list(twi_translations.keys())

# ==============================================================================
# REVERT TO THE ORIGINAL .KERAS MODEL FILE
# The .h5 detour is over.
# ==============================================================================
MODEL_PATH = "fine_tuned_model_3.0.keras"
FILE_ID = "1Zt6Fg4PeQx9WPIXXWzwQTP4FhczpZ3L9" # The very first ID we started with

def download_model():
    if not os.path.exists(MODEL_PATH):
        logger.info(f"📥 Downloading original .keras model...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
            logger.info("✅ Model downloaded.")
            return True
        except Exception as e:
            logger.error(f"❌ Error downloading model: {e}")
            return False
    else:
        logger.info(f"📁 Model file {MODEL_PATH} already exists")
    return True

def load_model_safely():
    if not download_model():
        return None
    try:
        logger.info(f"🔄 Attempting to load model with modern TensorFlow...")
        # No patches, no custom objects. Just a clean load.
        model = load_model(MODEL_PATH)
        logger.info("✅ Model loaded successfully! The correct environment is in place.")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}", exc_info=True)
        return None

# Image processing functions remain the same
def process_uploaded_image_pil(uploaded_file, img_size=(224, 224)):
    try:
        image = Image.open(uploaded_file.stream)
        if image.mode != 'RGB': image = image.convert('RGB')
        image = image.resize(img_size, Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"❌ Error processing image with PIL: {e}")
        return None

def process_uploaded_image(uploaded_file, img_size=(224, 224)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            class_dir = os.path.join(tmpdir, "class0")
            os.makedirs(class_dir, exist_ok=True)
            uploaded_file.seek(0)
            image_path = os.path.join(class_dir, uploaded_file.filename)
            with open(image_path, "wb") as f: f.write(uploaded_file.read())
            dataset = image_dataset_from_directory(
                tmpdir, image_size=img_size, batch_size=1, shuffle=False
            )
            for images, _ in dataset.take(1): return images.numpy()
    except Exception as e:
        logger.warning(f"⚠️ Dataset method failed: {e}. Falling back to PIL.")
        uploaded_file.seek(0)
        return process_uploaded_image_pil(uploaded_file, img_size)

# Initialize model at startup
logger.info(f"🚀 Initializing Twi Image Classifier...")
logger.info(f"🔧 TensorFlow version: {tf.__version__}")
model = load_model_safely()

if model is None:
    logger.error("💥 CRITICAL: Model could not be loaded!")
else:
    logger.info("🎉 VICTORY! Model initialized successfully!")
    model.summary(print_fn=logger.info)

# Flask routes remain the same
@app.route("/", methods=["GET", "POST"])
def index():
    if model is None: return render_template("index.html", predictions=None, warning="Model could not be loaded. Check console for details.")
    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("index.html", predictions=None, warning="Please upload an image.")
        image_file = request.files["image"]
        try:
            img_tensor = process_uploaded_image(image_file)
            if img_tensor is None: return render_template("index.html", predictions=None, warning="Error processing the uploaded image.")
            preds = model.predict(img_tensor, verbose=0)[0]
            top_indices = preds.argsort()[::-1][:3]
            top_preds = [(class_names[i], twi_translations.get(class_names[i], "❓"), float(preds[i])) for i in top_indices]
            logger.info(f"🎯 Top prediction: {top_preds[0][0]} ({top_preds[0][1]}) - {top_preds[0][2]:.1%}")
            if top_preds[0][2] >= 0.5:
                return render_template("index.html", predictions=top_preds, warning=None)
            else:
                return render_template("index.html", predictions=[], warning="Gyidie no nsɔ 50%, enti yɛrentumi nka (Confidence below 50%)")
        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}", exc_info=True)
            return render_template("index.html", predictions=None, warning="An error occurred while processing your image.")
    return render_template("index.html", predictions=None, warning=None)

@app.route("/health")
def health_check():
    return {"status": "healthy" if model is not None else "unhealthy", "model_loaded": model is not None, "tensorflow_version": tf.__version__, "classes_count": len(class_names)}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
