import json
import os
import sys

import torch
from chromadb import PersistentClient
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.join(os.pardir))
from project_consts import PROJECT_ROOT

device = "cuda" if torch.cuda.is_available() else "cpu"
papa = 1

data = "intents_prepared.json"  # intents data in data/ folder
model_tag = "multilingual-e5-large"  # tag to create collection in chromadb
embedding_model_name = "intfloat/multilingual-e5-large"  # model name from hf
embedding_model = HuggingFaceEmbeddings(
    cache_folder="/home/shared/models/sentence_transformers",
    model_name=embedding_model_name, 
    model_kwargs={"device": device},
)

if __name__ == "__main__":
    client = PersistentClient(os.path.join(PROJECT_ROOT, "chromadb_data"))
    collection = client.get_or_create_collection(model_tag)

    with open(os.path.join(PROJECT_ROOT, "data", data), encoding="utf-8") as intents_file:
        documents = json.load(intents_file)

    phrases = list(documents["phrase"].values())
    intents = list(documents["intent_path"].values())
    phrases_with_intents = list(zip(phrases, intents))

    embeddings = embedding_model.embed_documents(phrases)

    batch_size = 1000
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        metadata_batch = [
            {"text": phrase,
             "intent": intent} for phrase, intent in phrases_with_intents[i:i + batch_size]]
        ids = [f"phrase_{j}" for j in range(i, i + len(batch))]
        collection.add(embeddings=batch, ids=ids, metadatas=metadata_batch)

    query_result = collection.query(query_embeddings=embedding_model.embed_query("хочу осаго"), n_results=1)
    for i, doc in enumerate(query_result["documents"][0]):
        print("==============")
        print(i)
        print(query_result["distances"][0][i])
        print(query_result)
        print(doc)
        print("==============")