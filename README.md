# 😊 Multimodal Sentiment Analyzer  

A **full-stack web application** built with **Django** that analyzes sentiment in multiple modalities — text, images, and live webcam feeds.  
This project combines **RNN-LSTM based custom models**, **Google Gemini API**, and real-time processing to deliver accurate and interactive sentiment analysis.  

---

## 🚀 Features  

### Text Sentiment Analysis  
- Analyze the sentiment of user-provided text as **Positive, Neutral, or Negative**.  
- Trained on a **custom dataset** combining product reviews, Ebola-related tweets, and airline reviews.  
- Preprocessing includes advanced **text cleaning and tokenization**.  
- Provides **text description** using **Google Gemini API** to explain sentiment predictions.  

### File-based Batch Analysis  
- Upload a **CSV file** with multiple text entries.  
- Receive a **CSV output** with predicted sentiments and descriptions for each row.  

### Image Sentiment Analysis  
- Analyze **sentiment of images** using **Google Gemini API**.  
- Generates descriptive output explaining the detected sentiment.  

### Real-time Webcam Sentiment Analysis  
- Capture video from a webcam and analyze **facial expressions in real-time** to detect sentiment.  
- Provides instant **positive, neutral, or negative** predictions along with descriptive text.  

---

## 🛠 Tech Stack  
- **Backend:** Django  
- **AI / ML:** Python, TensorFlow (RNN + LSTM)  
- **Preprocessing:** Custom pipeline combining multiple datasets  
- **APIs:** Google Gemini API for descriptive outputs  
- **Frontend:** HTML, CSS, JS (Django templates)  
- **Data Handling:** CSV upload and download support  

