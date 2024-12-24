import gensim
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import json
import joblib
import os
import numpy as np

def load_or_cache_model(model_path, is_glove=False, save_model=False):
    model_dir = os.path.dirname(model_path)
    cache_path = os.path.join(model_dir, 'model.pkl')  

    if os.path.exists(cache_path):
        print(f"Loading cached model from {cache_path}")
        model = joblib.load(cache_path)
    else:
        print(f"Loading model from {model_path}")
        model = load_model(model_path, is_glove)
        if save_model:
            joblib.dump(model, cache_path)
    
    return model

def load_model(model_path, is_glove=False):
    if is_glove:
        return load_glove_model(model_path)
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model

def load_glove_model(glove_file):
    word_vectors = {}
    
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            vector = np.array([float(x) for x in tokens[1:]])
            word_vectors[word] = vector
    
    model = gensim.models.KeyedVectors(vector_size=len(next(iter(word_vectors.values()))))
    model.add_vectors(list(word_vectors.keys()), list(word_vectors.values()))
    return model

def load_config(filename):
    with open(filename, 'r') as f:
        return json.load(f)

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

def find_synonyms(word, model):
    try:
        synonyms = model.most_similar(word)
    except KeyError:
        print(f"'{word}' not found in the model vocabulary.")
        return {}
    
    def is_proper_noun(word):
        return word.istitle()
    
    def is_derivative(word, base_word):
        similarity = Levenshtein.ratio(word.lower(), base_word.lower())
        # print(f"Similarity between {word} and {base_word}: {similarity}")
        return similarity > 0.8 
    
    def multiword(word):
        return any(char in word for char in " -_")

    filtered_synonyms = {
        synonym: score for synonym, score in synonyms
        if not is_proper_noun(synonym) and not is_derivative(synonym, word) and not multiword(synonym)
    }

    return filtered_synonyms