import streamlit as st
import numpy as np
import pandas as pd
import txtai

def text_search():
    np.random.seed(1)
    df=pd.read_csv("train.csv")
    titles=df.dropna().sample(10000).TITLE.values

    embedding=txtai.Embeddings({'path' : 'sentence-transformers/all-MiniLM-L6-v2'})

    embedding.load("embedding.tar.gz")

    return titles , embedding

titles, embedding =st.cache_data(text_search)()
st.title("amazon product search")

query=st.text_input("enter the query")

if st.button("search"):
    if query:
        result=embedding.search(query, 5)
        actual_res=[titles[x[0]] for x in result]
        for res in actual_res:
            st.write(res)
    else:
      st.write("please enter a query")


