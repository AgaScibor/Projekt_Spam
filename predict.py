import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import numpy as np

# pobranie modelu oraz wekoryzacji 

with open ("models/model.pkl","rb") as f:
    model = pickle.load(f)
    
with open ("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
    
def predict_spam(text):
    X_input = vectorizer.transform([text])
    y_pred = model.predict(X_input)
    decision = model.decision_function(X_input)  
    probability = 1 / (1 + np.exp(-decision))   

    return "SPAM" if y_pred == 1 else "HAM", probability[0]