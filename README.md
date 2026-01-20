# 🛡️ Smart AI Spam Classifier (Self/online-Learning System)

Developed by **Papon**, future Machine Learning Engineer.

This is a professional-grade SMS/Email Spam Classifier built using **Machine Learning** and **Streamlit**. Unlike traditional static models, this app features **Online Learning** (Incremental Learning), allowing the model to improve itself based on real-time user feedback.

## 🚀 Live Demo
[(https://smssdpapon.streamlit.app/)]

## ✨ Key Features
* **Smart Prediction:** Accurately classifies messages as 'Spam' or 'Safe' using a trained SGDClassifier.
* **Incremental Learning:** Users can correct the model if it makes a mistake. The model learns from these corrections instantly without needing a full retrain.
* **Developer Dashboard:** A hidden Admin Panel (controlled via Sidebar) to track real-time analytics like total detections and Spam vs. Ham distribution.
* **Modern UI/UX:** Clean, dark-themed interface with a 'Clear Box' feature for seamless user experience.
* **Efficiency:** Uses `HashingVectorizer` for a memory-efficient and scalable text-to-numerical transformation.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Framework:** Streamlit
* **Libraries:** * `Scikit-Learn` (Machine Learning)
    * `NLTK` (Natural Language Processing)
    * `Pandas` & `NumPy` (Data Manipulation)
    * `Pickle` (Model Serialization)

## 📂 Project Structure
```text
├── app.py                 # Main Streamlit application code
├── hv_vectorizer.pkl      # Saved Hashing Vectorizer
├── online_model.pkl       # Saved SGDClassifier (Self-learning model)
├── stats.csv              # Storage for dashboard analytics
├── requirements.txt       # List of dependencies for deployment
└── README.md              # Project documentation
