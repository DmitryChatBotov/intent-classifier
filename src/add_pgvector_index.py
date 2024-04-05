import json
import os
import sys
import time

import torch
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector

sys.path.append(os.path.join(os.pardir))
from project_consts import PROJECT_ROOT

load_dotenv(find_dotenv())

device = "cuda" if torch.cuda.is_available() else "cpu"

data = "intents_prepared.json"  # intents data in data/ folder
model_tag = "multilingual-e5-large"
embedding_model_name = "intfloat/multilingual-e5-large"  # model name from hf
embedding_model = HuggingFaceEmbeddings(
    cache_folder="/home/shared/models/sentence_transformers",
    model_name=embedding_model_name, 
    model_kwargs={"device": device},
)

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PG_DRIVER", "psycopg2"),
    host=os.environ.get("PG_HOST"),
    port=int(os.environ.get("PG_PORT", "5432")),
    database=os.environ.get("PG_DB"),
    user=os.environ.get("PG_USER"),
    password=os.environ.get("PG_PASSWORD"),
)


if __name__ == "__main__":
    with open(os.path.join(PROJECT_ROOT, "data", data), encoding="utf-8") as intents_file:
        documents = json.load(intents_file)

    phrases = list(documents["phrase"].values())
    intents = list(documents["intent_path"].values())
    phrases_with_intents = list(zip(phrases, intents))

    from langchain.docstore.document import Document

    documents =  []

    for item in range(len(phrases_with_intents)):
        page = Document(
            page_content=phrases_with_intents[item][0], 
            metadata = {"intent": phrases_with_intents[item][1]}
        )
        documents.append(page)

    start_time = time.monotonic()
    db = PGVector.from_documents(
        embedding=embedding_model,
        documents=documents,
        collection_name=model_tag,
        connection_string=CONNECTION_STRING,
    )
    print(time.monotonic() - start_time)