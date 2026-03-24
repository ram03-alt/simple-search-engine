import numpy as np
import pandas as pd
import txtai 

np.random.seed(1)

df=pd.read_csv("train.csv")

# Sample 10,000 titles from the dataset, dropping any missing values
titles=df.dropna().sample(10000).TITLE.values

# Create an embedding model, index data, and save the model to disk
embedding=txtai.Embeddings({'path' : 'sentence-transformers/all-MiniLM-L6-v2'})

embedding.index(titles)

embedding.save("embedding.tar.gz")

#trial run for model test
# load model , and search similar title and choose top 5 similar titles
result=embedding.search("protector for cam", 5)

print(result)
#in result only index of title is stored, so we need to get the title from the index and store it in a list
ac_result=[titles[x[0]] for x in result]

print(ac_result)