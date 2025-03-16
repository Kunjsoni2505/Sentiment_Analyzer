import warnings
warnings.simplefilter("ignore", UserWarning)

import pandas as pd
import requests
import nltk
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
import google.generativeai as genai
import base64

# Prevent TensorFlow from using GPU (if running on CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


# Load the model and preprocessing pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'sentiment_analyzer', 'rnn_model_tf15.h5')
pre_path = os.path.join(BASE_DIR, 'sentiment_analyzer', 'text_pipeline.pkl')

with open(pre_path, 'rb') as file:
    loaded_pipeline = CustomUnpickler(file).load()

loaded_model = load_model(model_path, compile=False)

# Set up Gemini Pro API
GOOGLE_API_KEY = "Your-api-key"
genai.configure(api_key=GOOGLE_API_KEY)


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
    if isinstance(prediction, str):  # Check for error
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


# Function to process CSV input and return a CSV output
def process_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        uploaded_file = request.FILES['csv_file']
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        if 'text' not in df.columns:
            return HttpResponse("CSV must contain a 'text' column", status=400)

        # Initialize lists to hold results
        sentiments = []
        descriptions = []

        # Process each row in the DataFrame
        for text in df['text']:
            sentiment = predict_sentiment(text)
            description = get_description(text)
            sentiments.append(sentiment)
            descriptions.append(description)

        # Add the sentiment and description columns to the DataFrame
        df['sentiment'] = sentiments
        df['description'] = descriptions

        # Create a new CSV response to send to the user
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=sentiment_analysis_results.csv'

        writer = csv.writer(response)
        # Write the header
        writer.writerow(df.columns)
        # Write the data
        for row in df.values:
            writer.writerow(row)

        return response

    return render(request, 'upload_csv.html')  # A form for uploading CSV


def process_text(request):
    if request.method == 'POST':
        form = SentimentForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data['text']

            # Fetch sentiment and description
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


# image section---------------------------------------------------------------------------




def analyze_image(image_path):
    """Uses Gemini 1.5 Flash to analyze image sentiment and description separately."""
    try:
        image = Image.open(image_path)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = """Analyze the given image and provide a structured response. 
        1. Start with "Sentiment: " followed by a single-word or brief phrase (e.g., Happy, Sad, Neutral, Angry, etc.).
        2. Then, start a new line with "Description: " and provide a brief explanation of what the image depicts.
        """

        response = model.generate_content([prompt, image])

        # Default values
        sentiment = "Sentiment not detected"
        description = "No description available."

        if hasattr(response, "text") and response.text.strip():
            text = response.text.strip()

            # Extract sentiment using regex
            sentiment_match = re.search(r"Sentiment:\s*(.+)", text, re.IGNORECASE)
            if sentiment_match:
                sentiment = sentiment_match.group(1).strip()

            # Extract description using regex
            description_match = re.search(r"Description:\s*(.+)", text, re.IGNORECASE)
            if description_match:
                description = description_match.group(1).strip()

        return sentiment, description
    except Exception as e:
        return f"Error: {e}", "No description available."


def upload_image(request):
    """Handles image uploads, analyzes sentiment, and redirects to image result page."""
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
    """Receives frames from the webcam, detects faces, and analyzes sentiment."""
    if request.method == 'POST':
        try:
            image_data = request.POST.get('image')  # Expecting Base64 image

            if not image_data:
                return JsonResponse({'error': 'No image data received'}, status=400)

            # Decode Base64 image
            if "," in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                return JsonResponse({'error': 'Failed to decode image'}, status=400)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return JsonResponse({'sentiment': 'No face detected', 'description': 'No description available.'})

            # Extract the first detected face
            x, y, w, h = faces[0]
            face_img = img[y:y + h, x:x + w]

            # Save the face image temporarily
            image_path = default_storage.save('webcam_images/webcam_face.jpg',
                                              ContentFile(cv2.imencode('.jpg', face_img)[1].tobytes()))
            full_image_path = os.path.join(default_storage.location, image_path)

            # Analyze sentiment
            sentiment, description = analyze_image(full_image_path)

            return JsonResponse({'sentiment': sentiment, 'description': description})

        except Exception as e:
            return JsonResponse({'error': f'Server Error: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

def image_analysis(request):
    """Renders the image analysis page."""
    return render(request, 'image_analysis.html')

def image_result(request):
    """Renders the image result page with sentiment and description."""
    image_name = request.session.get('uploaded_image', '')  # Retrieve from session
    sentiment = request.session.get('sentiment', 'Unknown')
    description = request.session.get('description', 'No description available.')

    # Ensure correct image path
    image_path = settings.MEDIA_URL + image_name if image_name else ''

    return render(request, 'image_result.html', {
        'image_path': image_path,
        'sentiment': sentiment,
        'description': description
    })

def webcam_stream(request):
    """Renders the real-time webcam analysis page."""
    return render(request, 'webcam.html')
