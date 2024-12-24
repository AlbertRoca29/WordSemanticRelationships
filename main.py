import argparse
import sys
from utils import load_model, find_synonyms
from semantic_dimensions import get_semantic_dimensions, analyze_word_relationship

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Synonym Relationship Evaluator')
    parser.add_argument('word', type=str, help='Word to find synonyms for')
    parser.add_argument('--N', type=int, default=5, help='Number N')
    args = parser.parse_args()
    word, N = args.word, args.N

    

    # Load pre-trained model
    model_path = 'models/pretrained/word2vec/GoogleNews-vectors-negative300.bin.gz'
    model = load_model(model_path)

    synonyms = find_synonyms(word, model, N)

    if not synonyms:
        print(f"Could not find synonyms for '{word}'.")
        sys.exit(1)

    PROJECTION_THRESHOLD = 0.25
    
    print(f"Top {N} synonyms for '{word}':")
    for synonym, score in synonyms.items():
        print(f"{synonym}: {score}")

    
    config = {
        "gender": ["female", "male"], 
        "formality": ["formal", "casual"], 
        "plural": ["plural"]
    }

    semantic_dimensions = get_semantic_dimensions(model, config)

    for synonym in synonyms.keys():
        max_distance = 0
        max_dimension = None
        relationship = analyze_word_relationship(word, synonym, model, semantic_dimensions)
        for sd_name, sd in relationship.items():
            print(f"\nSemantic dimension: {sd_name}")
            print(f"Projection of '{word}': {sd['projection_word1']:.4f}")
            print(f"Projection of '{synonym}': {sd['projection_word2']:.4f}")
            print(f"Distance between projections: {sd['distance']:.4f}")
            if sd['distance'] > max_distance:
                max_distance = sd['distance']
                max_dimension = sd_name
        if max_distance >= PROJECTION_THRESHOLD:
            print(f"\nThe highest projection distance is in the '{max_dimension}' dimension with distance {max_distance:.4f}.")

if __name__ == "__main__":
    main()
