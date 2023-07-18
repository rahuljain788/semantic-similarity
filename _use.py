import tensorflow as tf
import tensorflow_hub as hub
from helpers import *
from download_data import *
import pandas as pd

# Load the pre-trained model
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    # Control GPU memory usage
    tf.config.experimental.set_memory_growth(gpu, True)

module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
model = hub.load(module_url)

# Generate Embeddings
# sentence_1 = stsb_test['sentence1']
# sentence_2 = stsb_test['sentence2']

sentence_3 = pd.Series("I remember asking you to pick DOT workstream task last week, what's the update on this")
sentence_4 = pd.Series("Yes, Am working on this will update you")
sentence_4 = pd.Series("Yes, i have started working on the DOT workstream task, will be available early next week, "
                       "will setup meeting with you on this")

# print(sentence_1.shape)
# print(sentence_2.shape)
print(sentence_3.shape)
print(sentence_4.shape)

sentence3_emb = model(sentence_3).numpy()
sentence4_emb = model(sentence_4).numpy()

# Cosine Similarity
# stsb_test['USE_cosine_score'] = cos_sim(sentence3_emb, sentence4_emb)

print(cos_sim(sentence3_emb, sentence4_emb))