from scipy.spatial import distance


def jaccard_similarity(y_true, y_pred):
    """Similarity score between two binary vectors (1 - Jaccard distance)."""
    return 1 - distance.jaccard(y_true, y_pred)
