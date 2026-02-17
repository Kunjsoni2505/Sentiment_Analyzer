# import warnings
# warnings.simplefilter("ignore", UserWarning)

import pandas as pd
import pickle
from .forms import SentimentForm
from .cleaner import TextCleaner, TextSequencer
from tensorflow.keras.models import load_model
from django.http import HttpResponse
import csv
import tensorflow as tf
from django.conf import settings
from django.shortcuts import render, redirect
import os
from django.http import JsonResponse
from PIL import Image
import cv2
import numpy as np
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import re
import base64
from dotenv import load_dotenv

from google import genai
import nltk
NLTK_DATA_PATH = "/opt/render/nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

for resource in ["wordnet", "stopwords", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

# Prevent TensorFlow from using GPU (if running on CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def home(request):
    return render(request, 'home.html')


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'TextCleaner':
            return TextCleaner
        elif name == 'TextSequencer':
            return TextSequencer
        return super().find_class(module, name)


# Load the model and preprocessing pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'sentiment_analyzer', 'rnn_model_tf15.h5')
pre_path = os.path.join(BASE_DIR, 'sentiment_analyzer', 'text_pipeline.pkl')

with open(pre_path, 'rb') as file:
    loaded_pipeline = CustomUnpickler(file).load()

loaded_model = load_model(model_path, compile=False)

# ===============================
# GEMINI SETUP (ONLY CHANGE)
# ===============================
load_dotenv()
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)


# Prediction function
def predict(text):
    try:
        processed_text = loaded_pipeline.transform(pd.Series([text]))
        prediction = loaded_model.predict(processed_text)
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


# ===============================
# UPDATED DESCRIPTION FUNCTION
# ===============================
def get_description(text):
    try:
        response = gemini_client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=f"Provide a brief description for: {text}"
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"


# Function to process CSV input and return a CSV output
def process_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        uploaded_file = request.FILES['csv_file']
        df = pd.read_csv(uploaded_file)

        if 'text' not in df.columns:
            return HttpResponse("CSV must contain a 'text' column", status=400)

        sentiments = []
        descriptions = []

        for text in df['text']:
            sentiments.append(predict_sentiment(text))
            descriptions.append(get_description(text))

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


def process_text(request):
    if request.method == 'POST':
        form = SentimentForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data['text']

            context = {
                'form': form,
                'prediction': predict_sentiment(data),
                'description': get_description(data),
            }

            return render(request, 'result.html', context)
    else:
        form = SentimentForm()

    return render(request, 'Text.html', {'form': form})


def result(request):
    return render(request, 'result.html')


# ===============================
# UPDATED IMAGE ANALYSIS (ONLY CHANGE)
# ===============================
def analyze_image(image_path):
    try:
        image = Image.open(image_path)

        prompt = """Analyze the given image and provide a structured response.
Sentiment: <short phrase>
Description: <brief explanation>
"""

        response = gemini_client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[prompt, image]
        )

        sentiment = "Sentiment not detected"
        description = "No description available."

        if response.text:
            text = response.text.strip()
            s = re.search(r"Sentiment:\s*(.+)", text, re.I)
            d = re.search(r"Description:\s*(.+)", text, re.I)

            if s:
                sentiment = s.group(1).strip()
            if d:
                description = d.group(1).strip()

        return sentiment, description

    except Exception as e:
        return f"Error: {e}", "No description available."


def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        image_path = default_storage.save(
            f'uploaded_images/{uploaded_image.name}',
            ContentFile(uploaded_image.read())
        )
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
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            ).detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return JsonResponse({'sentiment': 'No face detected', 'description': 'No description available.'})

            x, y, w, h = faces[0]
            face_img = img[y:y + h, x:x + w]

            image_path = default_storage.save(
                'webcam_images/webcam_face.jpg',
                ContentFile(cv2.imencode('.jpg', face_img)[1].tobytes())
            )
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
