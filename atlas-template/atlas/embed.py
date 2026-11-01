"""FIND: EmbeddingGemma 2, one open model on your own machine.

Text in any language, and pictures, land in one space of 768 numbers. A Nepali question sits close to an English
paragraph and to a photo of the same thing. Nothing is paid per use. The prompts are the ones the model card gives.
"""
import numpy as np
from PIL import Image

from .config import EMBED_MODEL

_model = None


def model():
    global _model
    if _model is None:                       # the first call downloads and loads it (once)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def query(text: str) -> np.ndarray:
    return model().encode("task: search result | query: " + text, normalize_embeddings=True)


def documents(texts: list[str], title: str = "none") -> np.ndarray:
    return model().encode([f"title: {title} | text: {t}" for t in texts], normalize_embeddings=True)


def picture(img: Image.Image) -> np.ndarray:
    """A photo goes in as an image, into the same space as the words."""
    return model().encode({"image": img.convert("RGB")}, normalize_embeddings=True)
