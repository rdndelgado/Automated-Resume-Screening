import pre_process as pr
import streamlit as st
from pathlib import Path
import spacy
from predict import predict_category
import time

file_uploads =[]
features = []
order = []
scores = []

def show_results(n, filename, score):
    with st.expander(f"See features for {filename}: {score}%"):
        skills = set()
        experiences = set()
        degree = set()

        for feature in features[order[n-1]]:

            if feature[-1] == 'Skills':
                skills.update(feature)
            elif feature[-1] == 'Experience':
                experiences.update(feature)
            elif feature[-1] == 'Degree':
                degree.update(feature)

        if skills:
            with st.container(border=True):
                st.write('Skills')
                st.write(skills)
        else:
            with st.container(border=True):
                st.write('Skills')
                st.write('None')
        if experiences:
            with st.container(border=True):
                st.write('Experiences')
                st.write(experiences)
        else:
            with st.container(border=True):
                st.write('Experiences')
                st.write('None')
        if degree:
            with st.container(border=True):
                st.write('Education')
                st.write(degree)
        else:
            with st.container(border=True):
                st.write('Education')
                st.write('None')
    
def main():

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    with st.container(border=True):
        st.title("Dynamic Resumé Engine (DRE)")
        with st.container(border=True):
            st.write("an automated Natural Language Processing and Machine Learning-driven Resume Screening tool for Faculty Recruitment at De La Salle-Dasmarinas.")

    tabs = st.tabs(["Home", "Results", "Category"])

    with tabs[0]:

        with st.container(border=True):

            st.header("Job Qualifications")
            job_description = st.file_uploader('Enter the job qualifications:', type="pdf", accept_multiple_files=False)

            jd = ''
            if job_description:
                jd = upload_dir / job_description.name
                with open(jd, "wb") as f:
                    f.write(job_description.read())
                job_description_text = pr.pdf_to_text(open(jd))

        with st.container(border=True):
            
            st.header("Resume")
            uploaded_files = st.file_uploader('Upload your resume(s) pdf file:', type="pdf", accept_multiple_files=True)

            st.write(f"Total pdf files uploaded: {len(uploaded_files)}")

            global file_uploads
            file_uploads = uploaded_files

            files = []
            if uploaded_files:
                for file in uploaded_files:
                    # Save the file to the "uploads" folder
                    file_path = upload_dir / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    files.append(file_path)

            pre_processed_resumes = []

            for file in files:
                pre_processed_resumes.append(pr.text_cleaning(pr.pdf_to_text(open(file))))

            extracted_features = []
            ner = Path('ner_model\ner')
            nlp = spacy.load(ner)

            global features

            for text in pre_processed_resumes:
                doc = nlp(text)
                features.append([(ent.text, ent.label_) for ent in doc.ents if ent])

                ents = ' '.join([ent.text for ent in doc.ents if ent])
                extracted_features.append(ents)
        with st.container(border=True):

            n_show = st.slider("Display top candidates", min_value=1, max_value=len(pre_processed_resumes))
            match = st.button("Compare!")

        with st.container(border=True):

            if match and job_description and files:

                progress_text = "Matching documents in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)

                top_n = pr.compute_word2vec_similarity(extracted_features, job_description_text)
                time.sleep(1)
                my_bar.empty()
                
                result = top_n

                job_title = pr.text_cleaning(job_description.name[:job_description.name.find('.pdf')]).upper()
                st.title(f'Top candidates for {job_title}')
                st.write(f'Remaining resumes: {len(pre_processed_resumes) - n_show}')
                st.write(f"{'-' * 3}")

                with st.container():

                    global order
                    for i, match_percentage in enumerate(top_n[:n_show]):
                        score = match_percentage[-1]

                        scores.append(score)
                        order.append(match_percentage[0])

                        st.write(
                            f'\t[{i + 1}] Filename: {file_uploads[match_percentage[0]].name} || score: {score}%')
            elif match:
                if not uploaded_files:
                    st.warning(
                        "Upload some files to compare or check the file format of the files you've uploaded for resumes.")
                elif not job_description:
                    st.error("Upload a job description file to compare.")

    with tabs[1]:
        st.header('Result')
        if len(order)>1:
            st.write('This sections shows the extracted experiences and skills from the resumes of top candidates.')
            #normal results
            for i in range(1, n_show+1):
                filename = file_uploads[order.copy()[i-1]].name
                show_results(i, filename, scores[i-1])
#-------------------------------------------------------------------------------------------#
            # st.subheader('Sorted by no. of skills present')
            # key = 'skills_len'
            # #will return an order list of sorted resumes by skills
            # by_categ_result_skills = sort_resumes_by_category(n_show, key)

            # for i in range(1, n_show+1):
            #     filename = file_uploads[by_categ_result_skills[i-1]].name
            #     show_results(i, filename, scores[i-1])
#-------------------------------------------------------------------------------------------#
            # st.subheader('Sorted by no. of experiences present')
            # key = 'exp_len'
            # #will return an order list of sorted resumes by experience
            # by_categ_result_exp = sort_resumes_by_category(n_show, key)

            # for i in range(1, n_show+1):
            #     filename = file_uploads[by_categ_result_exp[i-1]].name
            #     show_results(i, filename, scores[i-1])
#-------------------------------------------------------------------------------------------#
        elif len(order) == 1:
            filename = file_uploads[order[0]].name
            ents = features[order[0]]
            show_results(i, filename, scores[i], ents)

        else: st.write('Compare the resumes and Job qualifications first.')

    with tabs[2]:
        st.header('Category')
        st.write('This section shows the predicted category of the resumes based on their extracted features')
        if len(order)>1:
            for idx, top_resume in enumerate(result[:n_show]):
                resume = pre_processed_resumes[top_resume[0]]

                st.subheader(f"[{idx + 1}] {file_uploads[top_resume[0]].name}")
                st.write(f"Matched category: {predict_category(resume)}")
        else:
            st.write('No predicted category yet.')

if __name__ == "__main__":
    main()