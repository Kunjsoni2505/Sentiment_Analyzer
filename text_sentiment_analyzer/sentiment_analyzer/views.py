import warnings
warnings.simplefilter("ignore", UserWarning)

import pandas as pd
import requests
import nltk
import pickle
from .forms import SentimentForm
from .cleaner import TextCleaner, TextSequencer
from django.http import HttpResponse, JsonResponse
import csv
from django.conf import settings
from django.shortcuts import render, redirect
import os
from PIL import Image
import cv2
import numpy as np
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import re
import google.generativeai as genai
import base64

# Prevent TensorFlow from using GPU (CPU-only)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def home(request):
    return render(request, 'home.html')


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'TextCleaner':
            return TextCleaner
        elif name == 'TextSequencer':
            return TextSequencer
        return super().find_class(module, name)


# Load preprocessing pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pre_path = os.path.join(BASE_DIR, 'sentiment_analyzer', 'text_pipeline.pkl')
with open(pre_path, 'rb') as file:
    loaded_pipeline = CustomUnpickler(file).load()


# ---------------- Lazy-load TensorFlow model ---------------- #
_model = None

def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(BASE_DIR, 'sentiment_analyzer', 'rnn_model_tf15.h5')
        _model = tf.keras.models.load_model(model_path, compile=False)
    return _model
# ------------------------------------------------------------- #


# Set up Gemini Pro API
GOOGLE_API_KEY = "Your-api-key"
genai.configure(api_key=GOOGLE_API_KEY)


# ---------------- Prediction Functions ---------------- #
def predict(text):
    try:
        processed_text = loaded_pipeline.transform(pd.Series([text]))
        model = get_model()  # Lazy-load here
        prediction = model.predict(processed_text)
        return prediction
    except Exception as e:
        return str(e)


def predict_sentiment(text):
    prediction = predict(text)
    if isinstance(prediction, str):
        return f"Error: {prediction}"

    if prediction.shape[1] == 3:
        if prediction[0][0] > prediction[0][1] and prediction[0][0] > prediction[0][2]:
            return 'negative'
        elif prediction[0][1] > prediction[0][0] and prediction[0][1] > prediction[0][2]:
            return 'neutral'
        else:
            return 'positive'
    else:
        return "Prediction format not as expected."


def get_description(text):
    try:
        prompt = f"Provide a brief description for: {text}"
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text if hasattr(response, 'text') else "Error: No text generated."
    except Exception as e:
        return f"Error: {e}"
# ------------------------------------------------------- #


# ---------------- CSV Processing ---------------- #
def process_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        uploaded_file = request.FILES['csv_file']
        df = pd.read_csv(uploaded_file)

        if 'text' not in df.columns:
            return HttpResponse("CSV must contain a 'text' column", status=400)

        sentiments = []
        descriptions = []

        for text in df['text']:
            sentiment = predict_sentiment(text)
            description = get_description(text)
            sentiments.append(sentiment)
            descriptions.append(description)

        df['sentiment'] = sentiments
        df['description'] = descriptions

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=sentiment_analysis_results.csv'

        writer = csv.writer(response)
        writer.writerow(df.columns)
        for row in df.values:
            writer.writerow(row)

        return response

    return render(request, 'upload_csv.html')
# ------------------------------------------------- #


# ---------------- Text Processing ---------------- #
def process_text(request):
    if request.method == 'POST':
        form = SentimentForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data['text']
            sentiment = predict_sentiment(data)
            description = get_description(data)

            context = {
                'form': form,
                'prediction': sentiment,
                'description': description,
            }
            return render(request, 'result.html', context)
    else:
        form = SentimentForm()

    return render(request, 'Text.html', {'form': form})


def result(request):
    return render(request, 'result.html')
# ------------------------------------------------- #


# ---------------- Image Processing ---------------- #
def analyze_image(image_path):
    try:
        image = Image.open(image_path)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """Analyze the given image and provide a structured response. 
        1. Start with "Sentiment: " followed by a single-word or brief phrase (e.g., Happy, Sad, Neutral, Angry, etc.).
        2. Then, start a new line with "Description: " and provide a brief explanation of what the image depicts.
        """
        response = model.generate_content([prompt, image])

        sentiment = "Sentiment not detected"
        description = "No description available."

        if hasattr(response, "text") and response.text.strip():
            text = response.text.strip()
            sentiment_match = re.search(r"Sentiment:\s*(.+)", text, re.IGNORECASE)
            description_match = re.search(r"Description:\s*(.+)", text, re.IGNORECASE)
            if sentiment_match:
                sentiment = sentiment_match.group(1).strip()
            if description_match:
                description = description_match.group(1).strip()

        return sentiment, description
    except Exception as e:
        return f"Error: {e}", "No description available."


def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        image_path = default_storage.save(f'uploaded_images/{uploaded_image.name}', ContentFile(uploaded_image.read()))
        full_image_path = os.path.join(default_storage.location, image_path)
        sentiment, description = analyze_image(full_image_path)

        return render(request, 'image_result.html', {
            'image_path': image_path,
            'sentiment': sentiment,
            'description': description
        })

    return render(request, 'image_analysis.html')


def webcam_predict(request):
    if request.method == 'POST':
        try:
            image_data = request.POST.get('image')
            if not image_data:
                return JsonResponse({'error': 'No image data received'}, status=400)

            if "," in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                return JsonResponse({'error': 'Failed to decode image'}, status=400)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return JsonResponse({'sentiment': 'No face detected', 'description': 'No description available.'})

            x, y, w, h = faces[0]
            face_img = img[y:y + h, x:x + w]

            image_path = default_storage.save('webcam_images/webcam_face.jpg',
                                              ContentFile(cv2.imencode('.jpg', face_img)[1].tobytes()))
            full_image_path = os.path.join(default_storage.location, image_path)

            sentiment, description = analyze_image(full_image_path)

            return JsonResponse({'sentiment': sentiment, 'description': description})

        except Exception as e:
            return JsonResponse({'error': f'Server Error: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)


def image_analysis(request):
    return render(request, 'image_analysis.html')


def image_result(request):
    image_name = request.session.get('uploaded_image', '')
    sentiment = request.session.get('sentiment', 'Unknown')
    description = request.session.get('description', 'No description available.')
    image_path = settings.MEDIA_URL + image_name if image_name else ''

    return render(request, 'image_result.html', {
        'image_path': image_path,
        'sentiment': sentiment,
        'description': description
    })


def webcam_stream(request):
    return render(request, 'webcam.html')
