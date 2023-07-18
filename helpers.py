from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")


def text_processing(sentence):
    """
    Lemmatize, lowercase, remove numbers and stop words

    Args:
      sentence: The sentence we want to process.

    Returns:
      A list of processed words
    """
    sentence = [token.lemma_.lower()
                for token in nlp(sentence)
                if token.is_alpha and not token.is_stop]

    return sentence


def cos_sim(sentence1_emb, sentence2_emb):
    """
    Cosine similarity between two columns of sentence embeddings

    Args:
      sentence1_emb: sentence1 embedding column
      sentence2_emb: sentence2 embedding column

    Returns:
      The row-wise cosine similarity between the two columns.
      For instance is sentence1_emb=[a,b,c] and sentence2_emb=[x,y,z]
      Then the result is [cosine_similarity(a,x), cosine_similarity(b,y), cosine_similarity(c,z)]
    """
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)