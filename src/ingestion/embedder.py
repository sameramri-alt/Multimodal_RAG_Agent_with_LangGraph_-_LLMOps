from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from PIL import Image

class MultiModalEmbedder:
    """
    This class is responsible for transforming text and images into 'vectors' (embeddings).
    A vector is a long list of numbers that represents the mathematical meaning of the information.
    """
    def __init__(self):
        print("Loading CLIP Model (Dense)...")
        # The Dense Model: 'clip-ViT-B-32'
        # This model is incredible because it was trained on BOTH texts and images.
        # It projects both modalities into the EXACT SAME mathematical space.
        # Thus, the vector for the word "dog" will be very close to the vector of a picture of a dog.
        self.dense_model = SentenceTransformer('clip-ViT-B-32')
        
        print("Loading Sparse Model (FastEmbed BM25)...")
        # The Sparse Model: 
        # Unlike the dense model which understands the "global meaning", the sparse model focuses
        # on exact keyword matching (highly useful for searching proper nouns or specific technical terms).
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
    def embed_text_dense(self, text: str) -> list[float]:
        """
        Transforms a text (e.g., an audio transcript) into a Dense vector (its semantic meaning).
        """
        # Encode the text and convert it to a list of floats understandable by Qdrant.
        return self.dense_model.encode([text])[0].tolist()
        
    def embed_image_dense(self, image_path: str) -> list[float]:
        """
        Transforms an image (e.g., a presentation slide) into a Dense vector (its visual meaning).
        """
        # Open the image with Pillow (PIL) and encode it using the exact same model used for text.
        img = Image.open(image_path)
        return self.dense_model.encode([img])[0].tolist()
        
    def embed_text_sparse(self, text: str) -> dict:
        """
        Transforms a text into a Sparse vector (for keyword-based retrieval).
        """
        # Sparse encoding returns a special object, we take the first result.
        embeddings = list(self.sparse_model.embed([text]))[0]
        
        # For Qdrant, a sparse vector must be a dictionary containing two lists:
        # - indices: The mathematical ID of the words found.
        # - values: The importance (weight) of those words in the text.
        return {
            "indices": embeddings.indices.tolist(),
            "values": embeddings.values.tolist()
        }
