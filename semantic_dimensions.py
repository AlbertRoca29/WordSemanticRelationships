import numpy as np

class SemanticDimension:
    def __init__(self, model, dimension_words):
        self.model = model
        self.dimension_words = dimension_words
        self.vector = None
        self._compute_vector()

    def _compute_vector(self):
        if len(self.dimension_words) == 1:
            # Single-word definition (DEFINITION)
            self.vector = self.model[self.dimension_words[0]] + 1e-9
        elif len(self.dimension_words) == 2:
            # Word pair (CONTRAPOSITION)
            self.vector = self.model[self.dimension_words[0]] - self.model[self.dimension_words[1]]
        else:
            raise ValueError(f"Unexpected number of words for semantic dimension: {len(self.dimension_words)}")

        self.vector /= np.linalg.norm(self.vector)

    def get_vector(self):
        return self.vector

def get_semantic_dimensions(model, config):
    semantic_dimensions = {}
    for dimension, words in config.items():
        dimension_obj = SemanticDimension(model, words)
        semantic_dimensions[dimension] = dimension_obj.get_vector()
    return semantic_dimensions

def project_onto_semantic_dimension(word_vector, semantic_dimension):
    return np.dot(word_vector, semantic_dimension) * semantic_dimension

def analyze_word_relationship(word1, word2, model, SemanticDimensions):
    relationship = {}
    for sd_name, sd in SemanticDimensions.items():
        projection_word1 = project_onto_semantic_dimension(model[word1], sd)
        projection_word2 = project_onto_semantic_dimension(model[word2], sd)

        # Normalize projections
        projection_word1 = np.linalg.norm(projection_word1)
        projection_word2 = np.linalg.norm(projection_word2)

        relationship[sd_name] = {
            f'projection_{word1}': projection_word1,
            f'projection_{word2}': projection_word2,
            'distance': np.linalg.norm(projection_word1 - projection_word2)
        }
    return relationship

def compute_contraposition(model, positive_words, negative_words):
    positive_vector = np.mean([model[word] for word in positive_words], axis=0)
    negative_vector = np.mean([model[word] for word in negative_words], axis=0)
    contraposition = positive_vector - negative_vector
    return contraposition / np.linalg.norm(contraposition)
