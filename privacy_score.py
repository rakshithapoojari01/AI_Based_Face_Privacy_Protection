def calculate_privacy_score(similarity):
    """
    similarity: cosine similarity between original and protected face
    embeddings.
        +1  -> identical
         0  -> unrelated
        -1  -> completely opposite
 
    Maps the full theoretical range [-1, 1] linearly onto [0, 100]:
    similarity=1 (identical) -> 0% privacy, similarity=-1 (opposite)
    -> 100% privacy, similarity=0 (unrelated) -> 50% privacy.
 
    Note: for face verification specifically, real same-person pairs
    rarely exceed ~0.7 and typical match thresholds sit around 0.3-0.5,
    so this mapping is conservative in the practically-relevant range
    (a similarity of 0 -- already well past most "different person"
    thresholds -- only registers as 50%, not near-100%). Worth stating
    explicitly in your evaluation section since it's a real modeling
    choice, not an arbitrary constant.
    """
    similarity = max(0.0, min(1.0, similarity))
    privacy = (1.0 - similarity) * 100
    return round(privacy, 2)
 
 
def get_protection_level(score):
    """Centralized thresholds for the LOW / MEDIUM / HIGH label, so this
    logic lives in one place instead of being duplicated in a template.
    Matches what you've observed: ~54 -> MEDIUM, ~15-20 -> LOW.
    Adjust to match whatever your project report documents.
    """
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"
 
