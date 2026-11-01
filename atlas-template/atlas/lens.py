"""BIG IDEA 3, THE LENS: something the archive already knows, turned into a direction in the search space.

A lens is a vector (for example the average of pictures of one person, or a photo asked with the question).
Before searching, the question's own vector is pulled toward it, so the search finds that person or thing even
where no caption names it. No model is involved, only arithmetic on vectors that already exist.

In this template the lens is a photo attached to the question. Averaging the vectors of tagged pictures of a
person into an identity lens works the same way.
"""
import numpy as np

W_Q, W_L = 0.1, 0.9          # how much of the question stays, and how hard the lens pulls


def steer(q: np.ndarray, lenses: list[np.ndarray]) -> np.ndarray:
    if not lenses:
        return q
    v = W_Q * np.asarray(q) + W_L * np.mean(lenses, axis=0)
    return v / (np.linalg.norm(v) or 1.0)


def from_pictures(vectors: list[np.ndarray]) -> np.ndarray:
    """An identity lens: the average of pictures already tagged as one person, back to unit length."""
    v = np.mean(vectors, axis=0)
    return v / (np.linalg.norm(v) or 1.0)
