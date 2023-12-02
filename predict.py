import joblib
import spacy
from pre_process import text_cleaning, pdf_to_text
from pathlib import Path

def predict_category(resume_text):
    nlp = spacy.load('en_core_web_sm')
    knn = Path('knn_model.joblib')
    knn_model = joblib.load(knn)

    # Process the new resume text using spaCy and get its vector
    new_resume_vector = nlp(resume_text).vector

    # Reshape the vector to match the shape used during training
    new_resume_vector = new_resume_vector.reshape(1, -1)

    # Predict the category for the new resume
    predicted_category = knn_model.predict(new_resume_vector)

    return predicted_category[0]