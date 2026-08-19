import numpy as np


class PrivacyEvaluator:

    def __init__(self):

        self.weights = {
            "recognition": 0.40,
            "landmark": 0.20,
            "texture": 0.20,
            "coverage": 0.20
        }

    def recognition_privacy(self, similarity):
        similarity = np.clip(similarity, 0.0, 1.0)
        return (1 - similarity) * 100

    def landmark_score(self, mode):

        scores = {
            "natural": 40,
            "balanced": 65,
            "strong": 85,
            "maximum": 100
        }

        return scores.get(mode.lower(), 40)

    def texture_score(self, texture_enabled=True):

        if texture_enabled:
            return 90

        return 20

    def coverage_score(self, region_masks):

        total_regions = 6

        active = 0

        for mask in region_masks.values():

            if np.count_nonzero(mask) > 0:
                active += 1

        return (active / total_regions) * 100

    def evaluate(
        self,
        similarity,
        privacy_mode,
        region_masks,
        texture_enabled=True
    ):

        recognition = self.recognition_privacy(similarity)

        landmark = self.landmark_score(privacy_mode)

        texture = self.texture_score(texture_enabled)

        coverage = self.coverage_score(region_masks)

        privacy = (
            recognition * self.weights["recognition"] +
            landmark * self.weights["landmark"] +
            texture * self.weights["texture"] +
            coverage * self.weights["coverage"]
        )

        return {
            "similarity": round(similarity,4),
            "recognition_privacy": round(recognition,2),
            "privacy_score": round(privacy,2),
            "landmark_score": landmark,
            "texture_score": texture,
            "coverage_score": round(coverage,2)
        }