from django.shortcuts import render
import os
import pandas as pd
import requests
import nltk
import pickle
from .forms import SentimentForm
from .cleaner import TextCleaner, TextSequencer
import google.generativeai as genai  # Import the Gemini API library
from tensorflow.keras.models import load_model
from django.http import HttpResponse
import csv

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'TextCleaner':
            return TextCleaner
        elif name == 'TextSequencer':
            return TextSequencer
        return super().find_class(module, name)


# Load the model and preprocessing pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'sentiment_analyzer', 'rnn_model (3).h5')
pre_path = os.path.join(BASE_DIR, 'sentiment_analyzer', 'text_pipeline.pkl')

with open(pre_path, 'rb') as file:
    loaded_pipeline = CustomUnpickler(file).load()

loaded_model = load_model(model_path, compile=False)

# Set up Gemini Pro API
GOOGLE_API_KEY = "AIzaSyAz2e2mKPepUUkUWwQkoD41zCjcKqvjL0s"
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
        model = genai.GenerativeModel('gemini-pro')
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
