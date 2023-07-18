import openai
import os
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from helpers import *
from download_data import *
import pandas as pd

openai.api_key = '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'

sentence_3 = pd.Series("I remember asking you to pick DOT workstream task last week, what's the update on this")
sentence_4 = pd.Series("Yes, i have started working on the mapping task, will be available early next week, "
                       "will setup meeting with you on this")

if os.path.exists('data/nlp/davinci_emb.pkl'):
    print('Loading Davinci Embeddings')
    with open('data/nlp/davinci_emb.pkl', 'rb') as f:
        davinci_emb = pickle.load(f)
else:
    print('Querying Davinci Embeddings')
    davinci_emb = {}
    engine='text-similarity-davinci-001'

    unique_sentences = list(set(sentence_3.values.tolist() + sentence_4.values.tolist()))
    for sentence in tqdm(unique_sentences):
        if sentence not in davinci_emb.keys():
            davinci_emb[sentence] = openai.Embedding.create(input = [sentence],
                                                            engine=engine)['data'][0]['embedding']
    # Save embeddings to file
    with open('data/nlp/davinci_emb.pkl', 'wb') as f:
        pickle.dump(davinci_emb, f)

# Generate Embeddings
sentence1_emb = [davinci_emb[sentence] for sentence in sentence_3]
sentence2_emb = [davinci_emb[sentence] for sentence in sentence_4]

# Cosine Similarity
print(cos_sim(sentence1_emb, sentence2_emb))
# stsb_test['OpenAI Davinci_cosine_score'] = cos_sim(sentence1_emb, sentence2_emb)