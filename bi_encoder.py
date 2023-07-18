from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
from helpers import *
from download_data import *
import pandas as pd

# Load the pre-trained model
model = SentenceTransformer('stsb-mpnet-base-v2')

sentence_3 = pd.Series("I remember asking you to pick DOT workstream task last week, what's the update on this")
sentence_4 = pd.Series("Yes, i have started working on the mapping task, will be available early next week, "
                       "will setup meeting with you on this")

# Generate Embeddings
sentence1_emb = model.encode(sentence_3, show_progress_bar=True)
sentence2_emb = model.encode(sentence_4, show_progress_bar=True)

# Cosine Similarity
print(cos_sim(sentence1_emb, sentence2_emb))
# stsb_test['SBERT BiEncoder_cosine_score'] = cos_sim(sentence1_emb, sentence2_emb)