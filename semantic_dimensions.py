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


def compute_contraposition(model, word_pairs):
    word_pairs = np.array(word_pairs)
    assert word_pairs.shape[1] == 2, "Input array must have two columns (positive, negative)"
    contraposition = np.mean([(model[pos] - model[neg])/np.linalg.norm(model[pos] - model[neg]) for pos, neg in word_pairs], axis=0)
    return contraposition/np.linalg.norm(contraposition)


def compute_semantic_dimensions(model, data):
    semantic_dimensions = {}

    for dimension, values in data.items():
        contraposition = compute_contraposition(model, values)
        semantic_dimensions[dimension] = contraposition

    return semantic_dimensions
