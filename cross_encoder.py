from sentence_transformers import CrossEncoder
import tensorflow as tf
import tensorflow_hub as hub
from helpers import *
from download_data import *
import pandas as pd

# Load the pre-trained model
model = CrossEncoder('cross-encoder/stsb-roberta-base')

sentence_3 = pd.Series("I remember asking you to pick DOT workstream task last week, what's the update on this")
sentence_4 = pd.Series("Yes, i have started working on the DOT workstream task, will be available early next week, "
                       "will setup meeting with you on this")
sentence_pairs = []
for sentence1, sentence2 in zip(sentence_3, sentence_4):
    sentence_pairs.append([sentence1, sentence2])


print(model.predict(sentence_pairs, show_progress_bar=True))
# stsb_test['SBERT CrossEncoder_score'] = model.predict(sentence_pairs, show_progress_bar=True)
