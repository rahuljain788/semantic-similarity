import textdistance
from helpers import *
from download_data import *

def jaccard_sim(row):
    # Text Processing
    sentence1 = text_processing(row['sentence1'])
    sentence2 = text_processing(row['sentence2'])
    print(sentence1, sentence2)

    # Jaccard similarity
    return textdistance.jaccard.normalized_similarity(sentence1, sentence2)


# Jaccard Similarity
stsb_test['Jaccard_score'] = stsb_test.progress_apply(jaccard_sim, axis=1)

print(stsb_test['Jaccard_score'])