import joblib
import spacy
from pre_process import text_cleaning, pdf_to_text

def predict_category(resume_text):
    nlp = spacy.load('en_core_web_sm')
    knn_model = joblib.load('knn_model.joblib')

    # Process the new resume text using spaCy and get its vector
    new_resume_vector = nlp(resume_text).vector

    new_resume_vector = new_resume_vector.reshape(1, -1)

    predicted_category = knn_model.predict(new_resume_vector)

    return predicted_category[0]
