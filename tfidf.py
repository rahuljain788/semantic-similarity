from helpers import *
from download_data import *

from sklearn.feature_extraction.text import TfidfVectorizer
model = TfidfVectorizer(lowercase=True, stop_words='english')

# Train the model
X_train = pd.concat([stsb_train['sentence1'], stsb_train['sentence2']]).unique()
model.fit(X_train)

# Generate Embeddings on Test
sentence1_emb = model.transform(stsb_test['sentence1'])
sentence2_emb = model.transform(stsb_test['sentence2'])

# Cosine Similarity
stsb_test['TFIDF_cosine_score'] = cos_sim(sentence1_emb, sentence2_emb)
print(stsb_test['TFIDF_cosine_score'])