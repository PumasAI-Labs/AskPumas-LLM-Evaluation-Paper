from dotenv import dotenv_values
import os
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm 

API_KEY = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))
client = OpenAI(api_key = API_KEY["OPENAI_API_KEY"])

def get_embedding(text, model = 'text-embedding-3-small'):
    """
    This function allows us to input text and obtain embeddings using OpenAI's text-embedding-3-small model.
    Return the embedding vector, if an error occurs, it returns None.
    """
    if pd.isna(text):  
        return None
    text = text.replace("\n", " ")  
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"{e}")
        return None 

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "multillm.csv"))
embedding_df = pd.DataFrame(columns=df.columns)

for column in df.columns:
    print(f"Processing column: {column}")  
    embedding_df[column] = [
        get_embedding(x) for x in tqdm(df[column], desc=f'Generating embeddings for {column}')
    ]

embedding_df.to_csv(os.path.join(os.path.dirname(__file__), "multillm-embedd.csv"), index=False)

