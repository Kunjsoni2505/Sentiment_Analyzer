# 🧠 Multimodal Sentiment Intelligence Platform

🌐 **Live Application:**  
https://sentiment-analyzer-2h3q.onrender.com

A full-stack **AI-powered sentiment analysis platform** capable of analyzing sentiment from **text, CSV files, images, and live webcam streams**, combining **deep learning models** with **generative AI explanations**.

---

## 🚀 Project Overview

This project is an **end-to-end multimodal AI system** built using **Python, Django, TensorFlow (LSTM)**, and the **Gemini API**.  
Unlike basic sentiment projects, this platform focuses on **real-world deployment challenges**, **resource constraints**, and **production stability**.

The system not only predicts sentiment but also provides **contextual, human-readable explanations**, improving interpretability and user trust.

---

## ✨ Key Features

### 🔤 Text Sentiment Analysis
- Real-time sentiment prediction from raw text
- Outputs: **Negative / Neutral / Positive**
- Gemini API generates contextual explanations

### 📂 CSV Sentiment Analysis
- Upload CSV files containing a `text` column
- Batch sentiment inference
- Download processed CSV with results

### 🖼️ Image Sentiment Analysis
- Analyze sentiment from uploaded images
- Uses Gemini Vision capabilities
- Returns **sentiment + short description**

### 🎥 Webcam Sentiment Detection
- Live webcam capture
- Face detection using OpenCV
- Image-based sentiment inference

### 🤖 Generative AI Integration
- Gemini API enhances predictions with explanations
- Adds semantic understanding beyond classification

---

## 🧠 Machine Learning Pipeline

### Dataset
- Aggregated from multiple public text sources
- Balanced across sentiment classes
- Reduced class bias for better generalization

### Text Processing
- Custom preprocessing pipeline:
  - Cleaning
  - Tokenization
  - Sequencing
- Serialized using `pickle` for production reuse

### Model Architecture
- RNN–LSTM network with dropout regularization
- Built using TensorFlow / Keras
- Optimized for CPU inference

### Performance
- Achieved **85%+ accuracy**
- Stable inference under constrained memory environments

---

## 🛠 Tech Stack

### Backend
- Python
- Django
- Django CORS Headers

### Machine Learning & NLP
- TensorFlow (CPU)
- scikit-learn
- NLTK
- NumPy, Pandas

### Computer Vision
- OpenCV
- Pillow

### Generative AI
- Gemini API (`google-genai`)

### Deployment & Infrastructure
- Render
- Gunicorn
- Whitenoise
- Environment-based configuration

---

## ⚙️ Production Challenges Addressed

- ✔ Deploying TensorFlow models under limited RAM
- ✔ Preventing Gunicorn worker crashes
- ✔ Managing cold starts and long inference times
- ✔ Handling Gemini API rate limits
- ✔ Secure environment variable management
- ✔ Static & media file handling in production

This project intentionally explores **real deployment constraints**, not just local ML execution.
