import gensim
from sklearn.metrics.pairwise import cosine_similarity

def load_model(model_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model

def find_synonyms(word, model, top_n=10):
    try:
        synonyms = model.most_similar(word, topn=top_n)
        return dict(synonyms)
    except KeyError:
        print(f"'{word}' not found in the model vocabulary.")
        return {}

def compute_cosine_similarity(word1, word2, model):
    try:
        vec1, vec2 = model[word1], model[word2]
        return cosine_similarity([vec1], [vec2])[0][0]
    except KeyError as e:
        print(f"Error with words: {e}")
        return None

def compute_similarities(word, synonyms, model):
    similarities = {}
    for synonym, _ in synonyms.items():
        similarities[synonym] = compute_cosine_similarity(word, synonym, model)
    return similarities