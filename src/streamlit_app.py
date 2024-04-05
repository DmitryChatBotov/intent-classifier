import os
import sys

import pandas as pd
import streamlit as st
from chromadb import PersistentClient
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.join(os.pardir))
from project_consts import PROJECT_ROOT
from src.levenshtein_model import LevenshteinClassifier

load_dotenv(find_dotenv())

st.set_page_config(layout="wide")
models = [
    {
        "name": "intfloat/multilingual-e5-large",
        "type": "hf",
        "tag": "multilingual-e5-large",
        "min": 0.0,
        "max": 1.0,
        "default_threshold": 0.2
    },
    {
        "name": "Levenshtein Classifier",
        "tag": "levenshtein",
        "min": 0,
        "max": 15,
        "default_threshold": 5
    }
]
DATA_NAME = "intents_prepared.json"

chromadb_client = PersistentClient(os.path.join(PROJECT_ROOT, "chromadb_data"))

def initialize_models(models):
    for model in models:
        if model.get("type") == "hf":
            model["model"] = HuggingFaceEmbeddings(
                model_name=model["name"], 
                model_kwargs={"device": "cuda"},
                cache_folder=os.environ["CACHE_FOLDER"],
            )
            model["chroma_collection"] = chromadb_client.get_collection(model["tag"])
        else:
            model["model"] = LevenshteinClassifier(pd.read_json(os.path.join(PROJECT_ROOT, "data", DATA_NAME)))

initialize_models(models)
st.title("Классификация намерения (интента) пользователя")

thresholds = {}
cols = st.columns(len(models))
for i, model in enumerate(models):
    with cols[i]:
        model_name = model["name"]
        st.markdown(f"**{model_name}**")
        thresholds[model_name] = st.number_input(
            f"Порог по близости (от {model['min']} до {model['max']})",
            min_value=model["min"],
            max_value=model["max"],
            value=model["default_threshold"],
            key=model_name
        )

user_input = st.text_input("Введите вопрос:")
submit_button = st.button("Классифицировать")

if (submit_button and user_input) or user_input:
    cols = st.columns(len(models))
    
    for i, model in enumerate(models):
        with cols[i]:
            if model.get("type") == "hf":
                embedding_model = model["model"]
                collection = model["chroma_collection"]
                query_result = collection.query(query_embeddings=embedding_model.embed_query(user_input), n_results=5)
                print(query_result)
                
                distance = query_result["distances"][0][0]
                intent = query_result["metadatas"][0][0]["intent"]
                phrase =  query_result["metadatas"][0][0]["text"]
                is_close = distance <= thresholds[model["name"]]

            else:
                levenshtein_model = model["model"]
                intent, phrase, distance = levenshtein_model.classify_question(user_input)
                is_close = distance <= thresholds[model["name"]]

            if is_close:
                st.write(f"Интент: {intent}")
                st.write(f"Схожая фраза: {phrase}")
                st.write(f"Расстояние: {distance}")
            else:
                st.write("Интент не найден. Вопрос для генеративки")

