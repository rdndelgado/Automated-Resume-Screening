import pre_process as pr
import streamlit as st
import os
import spacy
from predict import predict_category


result = []
file_uploads = []

def main():
    global result

    st.title("Dynamic Resumé Engine (DRE)")
    desc = "an automated Natural Language Processing and Machine Learning-driven Resume Screening tool for Faculty Recruitment at De La Salle-Dasmarinas."
    st.write(desc)
    # Corrected usage of st.tabs
    tabs = st.tabs(["Home", "Results", "Category"])
    
     #storage of uploaded files
    os.makedirs("uploads", exist_ok=True)

    with tabs[0]:  # Accessing the first tab
        st.write('Note: The app only accepts pdf file formats for upload, otherwise it will be disregarded.')
        st.header("Resume")

        uploaded_files = st.file_uploader('Upload your resume(s) pdf file:', type="pdf", accept_multiple_files=True)
        len_uploads = len([res for res in uploaded_files if res])

        st.write(f"Total pdf files uploaded: {len_uploads}")
        global file_uploads
        file_uploads = uploaded_files

        files = []
        if uploaded_files:
            for file in uploaded_files:
                # Save the file to the "uploads" folder
                file_path = os.path.join("uploads", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())
                files.append(file_path)

        pre_processed_resumes = []

        for file in files:
            pre_processed_resumes.append(pr.text_cleaning(pr.pdf_to_text(open(file))))

        extracted_features = []
        nlp = spacy.load('ner_model')

        for text in pre_processed_resumes:
            doc = nlp(text)
            ents = ' '.join([ent.text for ent in doc.ents if ent])
            extracted_features.append(ents)

        st.header("Job Qualifications")
        job_description = st.file_uploader('Enter the job qualifications:',type="pdf", accept_multiple_files=False)
        
        jd = ''
        if job_description:

            jd = os.path.join("uploads", job_description.name)
            with open(jd, "wb") as f:
                f.write(job_description.read())
            job_description_text = pr.pdf_to_text(open(jd))

        n_show = st.number_input(f"Show top candidates:", step=1, value=1, format="%d")  
        match = st.button("Compare!")

        if match and job_description and files:

            top_n = pr.compute_word2vec_similarity(extracted_features, job_description_text)
            result = top_n

            job_title = pr.text_cleaning(job_description.name[:job_description.name.find('.pdf')]).upper()
            st.title(f'Top candidates for {job_title}')
            st.write(f'Remaining resumes: {len_uploads-n_show}')
            st.write(f"{'-'*11}")

            for i, match_percentage in enumerate(top_n[:n_show]):
                score = match_percentage[-1]
                st.write(f'\t[{i+1}] Filename: {file_uploads[match_percentage[0]].name} || score: {score}%')
        else:
            if not uploaded_files:
                st.write("Upload some files to compare or check the file format of the files you've uploaded for resumes.")
            elif not job_description:
                st.write("Upload a job description file to compare.")

    with tabs[1]:  # Accessing the second tab
        st.header("Results")

    with tabs[2]:
        st.header('Predicted Category')

        for idx, top_resume in enumerate(result[:n_show]):
            resume = pre_processed_resumes[top_resume[0]]

            st.subheader(f"[{idx+1}] {file_uploads[top_resume[0]].name}")
            st.write(f"Matched category: {predict_category(resume)}")
        
if __name__ == "__main__":
    main()
