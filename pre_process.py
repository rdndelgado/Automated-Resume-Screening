from nltk import word_tokenize
from nltk.corpus import stopwords
import re
import fitz
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def pdf_to_text(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    
    for page_number in range(doc.page_count):
        page = doc[page_number]
        text += page.get_text()

    doc.close()
    return text

def text_cleaning(text, type=str):
    if type == 'degree':
        keywords = ['city', 'state', 'usa', 'university', 'college', 'school','month', 'nyack', 'Iowa', 'gpa']
        for keyword in keywords:
            text = re.sub(rf'\b{re.escape(keyword)}\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    else:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    doc = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    removed_sw = [token for token in doc if token not in stop_words]
    text = ' '.join(removed_sw)
    return text

def compute_word2vec_similarity(resumes_features, job_description):
    """
    Parametes:
        resumes (list): list of pre-processed text of resumes
        job_description (str): text format of job description
    """
    # cleaned_resumes = []
    # nlp = spacy.load('ner_model')
    # for resume in resumes:
    #     doc = nlp(resume)
    #     ents = ' '.join([ent.text for ent in doc.ents if ent])
    #     cleaned_resumes.append(ents)

    data_jd = text_cleaning(job_description)

    tokenized_jd = data_jd.split()
    tokenized_resumes = [resume.split() for resume in resumes_features]

    model = Word2Vec([tokenized_jd] + tokenized_resumes, vector_size=100, window=5, min_count=1, workers=4)

    # Average the word vectors for each document
    avg_vectors = []
    for tokens in [tokenized_jd] + tokenized_resumes:
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if vectors:
            avg_vector = sum(vectors) / len(vectors)
            avg_vectors.append(avg_vector)
        else:
            # If no words are found in the Word2Vec model, use a zero vector
            avg_vectors.append([0] * model.vector_size)

    # Compute cosine similarity for each resume
    match_percentages = []
    for i in range(1, len(resumes_features) + 1):
        match_percentage = cosine_similarity([avg_vectors[0]], [avg_vectors[i]])[0][0] * 100
        match_percentage = round(match_percentage, 2)
        match_percentages.append((i-1, match_percentage))
    
    match_percentages = sorted(match_percentages, key=lambda x: x[-1], reverse=True)

    return match_percentages