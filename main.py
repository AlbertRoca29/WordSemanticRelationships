import argparse
import sys
from utils import load_or_cache_model, find_synonyms, load_config
from semantic_dimensions import analyze_word_relationship, compute_semantic_dimensions 
import time

def main():
    save_model = False # Set to True if you want to save the model and have faster experiments

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Synonym Relationship Evaluator')
    parser.add_argument('word', type=str, help='Word to find synonyms for')
    args = parser.parse_args()
    word = args.word
    
    PROJECTION_THRESHOLD = 0.8
    SIMILARITY_THRESHOLD = 0.75

    # Load pre-trained model
    model_path = 'models/pretrained/Glove/glove.42B.300d.txt'
    a = time.time()
    model = load_or_cache_model(model_path, save_model=True, is_glove=True)
    print("Time taken to load model: ", time.time()-a)
    
    config_file = 'semantic_dimensions_config.json'
    config = load_config(config_file)

    words = [word]
    # words = [
    #     "apple", "book", "car", "chair", "dog", "cat", "house", "computer", "tree", "table", 
    #     "phone", "flower", "person", "window", "building", "road", "city", "mountain", "ocean", 
    #     "river", "shoe", "shirt", "food", "water", "sun", "moon", "light", "air", "cloud", "music", 
    #     "song", "movie", "game", "coffee", "school", "university", "child", "friend", "family", 
    #     "work", "idea", "problem", "solution", "question", "answer", "love", "hate", "peace", 
    #     "joy", "time", "horizon", "whisper", "echo", "tunnel", "flame", "concept", "illusion", 
    #     "paradox", "serenity", "vision", "mystery", "glimpse", "magnitude", "vortex"
    # ]

    semantic_dimensions = compute_semantic_dimensions(model, config)

    for word in words:
        print(f"\nWord: {word}")
        synonyms = find_synonyms(word, model)

        if not synonyms:
            # print(f"Could not find synonyms for '{word}'.")
            # sys.exit(1)
            ...
        
        for synonym, score in synonyms.items():
            if score < SIMILARITY_THRESHOLD:
                continue
            relationship = analyze_word_relationship(word, synonym, model, semantic_dimensions)
            SD = []
            for sd_name, sd in relationship.items():
                if sd['distance'] >= PROJECTION_THRESHOLD:
                    SD.append(sd_name)
                    # print(f"\nSemantic dimension: {sd_name}")
                    # print(f"Projection of '{word}': {sd[f'projection_{word}']:.4f}")
                    # print(f"Projection of '{synonym}': {sd[f'projection_{synonym}']:.4f}")
                    # print(f"Distance between projections: {sd['distance']:.4f}")
            if SD:
                print(f"{synonym} ({', '.join(SD)})")
            else:
                print(f"{synonym}")
        print("_" * 35)
    # for synonym in synonyms.keys():
    #     max_distance = 0
    #     max_dimension = None
    #     relationship = analyze_word_relationship(word, synonym, model, semantic_dimensions)
    #     for sd_name, sd in relationship.items():
    #         print(f"\nSemantic dimension: {sd_name}")
    #         print(f"Projection of '{word}': {sd['projection_word1']:.4f}")
    #         print(f"Projection of '{synonym}': {sd['projection_word2']:.4f}")
    #         print(f"Distance between projections: {sd['distance']:.4f}")
    #         if sd['distance'] > max_distance:
    #             max_distance = sd['distance']
    #             max_dimension = sd_name
    #     if max_distance >= PROJECTION_THRESHOLD:
    #         print(f"\nThe highest projection distance is in the '{max_dimension}' dimension with distance {max_distance:.4f}.")

if __name__ == "__main__":
    main()
