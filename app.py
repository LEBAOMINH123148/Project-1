import streamlit as st
from sentence_transformers import SentenceTransformer, util
import whisper
import os

st.title("OHaiYo :red[ONi-chaan]", text_alignment="center")


# modelw for transcribe audio to text, models for tranform sentence to vector
@st.cache_resource
def initModel():
    modelw = whisper.load_model("small")
    models = SentenceTransformer("all-MiniLM-L6-v2")
    return models, modelw


# The Ear, return audio after transcribed
@st.cache_data
def process_uploadedfile(_modelw, file_path, file_name):
    result = modelw.transcribe(file_path)
    user_data = []
    for segment in result["segments"]:
        item = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
        }
        user_data.append(item)
    return user_data


# load data and embedd it
@st.cache_data
def Loaddata(_models, user_data):
    sentences_to_check = []
    for segment in user_data:
        sentences_to_check.append(segment["text"])
    embedding = models.encode(sentences_to_check)
    return embedding


# The Brain
def Working(user_input, user_data, models, embedding, uploaded_file):
    user_input = models.encode(user_input.lower())
    result = util.semantic_search(user_input, embedding, top_k=3)
    for i in result[0]:
        id = i["corpus_id"]
        score = i["score"] * 100
        text = user_data[id]["text"]
        time_start = user_data[id]["start"]
        st.write(
            f"Time start: {time_start:.1f}s, Sentence {id}: '{text}' Percentage: {score:.1f}%\n"
        )
        st.audio(uploaded_file, format="audio/wav", start_time=time_start)


uploaded_file = st.file_uploader("Upload your file here(Video or audio): ")
if uploaded_file:
    with open("temp_uploaded_file", "wb") as f:
        f.write(
            uploaded_file.getvalue()  # create temp file that has content of the user uploaded file to put it in whisper
        )
    st.success("File Uploaded")
    models, modelw = initModel()
    user_data = process_uploadedfile(
        modelw, "temp_uploaded_file", uploaded_file.name
    )  # whisper need file path not file on ram (st.file_upload store data on ram)
    embedding = Loaddata(models, user_data)
    os.remove("temp_uploaded_file")

    with st.form(key="search_form"):
        query = st.text_input("What do you want to find?: ")
        submit_button = st.form_submit_button("Search")

        if submit_button:
            if query:
                Working(query, user_data, models, embedding, uploaded_file)
            else:
                st.warning("Please tell us what you want to find first")
