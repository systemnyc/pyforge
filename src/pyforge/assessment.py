"""Compare student explanations with reference explanations using embeddings."""
from pyforge.retrieval import embedding_model
from sklearn.metrics.pairwise import cosine_similarity


def score_explanation(explanation: str, reference: str) -> float:

    """
    Calculate semantic similarity between two explanations.

        Args:
            explanation: The student's explanation in their own words.
            reference: The reference explanation used for comparison.

        Returns:
            Cosine similarity between the text embeddings, generally ranging
            from -1 to 1. Higher values indicate greater semantic similarity.

        Note:
            This score contributes evidence toward understanding assessment.
            It must be combined with correctness checks and practice results
            before making a mastery decision.
    """
    vectors = embedding_model.embed_documents([explanation, reference])
    similarity = cosine_similarity([vectors[0]], [vectors[1]])

    return float(similarity[0][0])