from django.shortcuts import render
import os
import pandas as pd
from django.shortcuts import render
from .forms import SentimentForm
import nltk
import pickle

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from .cleaner import TextCleaner, TextSequencer
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'TextCleaner':
            return TextCleaner
        elif name == 'TextSequencer':
            return TextSequencer
        return super().find_class(module, name)
# # Load the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'sentiment_analyzer', 'rnn_model (3).h5')
pre_path = os.path.join(BASE_DIR, 'sentiment_analyzer', 'text_pipeline.pkl')
# model = load_model(model_path, compile=False)


# # Load the pipeline
with open(pre_path, 'rb') as file:
    loaded_pipeline = CustomUnpickler(file).load()

# Load the model
from tensorflow.keras.models import load_model
loaded_model = load_model(model_path, compile = False)


# Prediction function
def process_text(request):
    if request.method == 'POST':
        form = SentimentForm(request.POST)
        if form.is_valid():
            # Retrieve cleaned data from the form
            data = form.cleaned_data['text']
            def predict(text):
                processed_text = loaded_pipeline.transform(pd.Series([text]))
                prediction = loaded_model.predict(processed_text)
                return prediction
            def predict_sentiment(text):
                prediction = predict(text)
                # Check the shape of prediction and handle it accordingly
                if prediction.shape[1] == 3:
                    sentiment = ''
                    if prediction[0][0] > prediction[0][1] and prediction[0][0] > prediction[0][2]:
                        sentiment = 'negative'
                    elif prediction[0][1] > prediction[0][0] and prediction[0][1] > prediction[0][2]:
                        sentiment = 'neutral'
                    else:
                        sentiment = 'positive'
                    return sentiment
                else:
                    return "Prediction format not as expected."
            context = {
                'form': form,
                'prediction': predict_sentiment(data),
                # Include input data for rendering in the template
            }

            return render(request, 'result.html', context)
    else:
        form = SentimentForm()

    return render(request, 'Text.html', {'form': form})


def result(request):
    return render(request, 'result.html')
