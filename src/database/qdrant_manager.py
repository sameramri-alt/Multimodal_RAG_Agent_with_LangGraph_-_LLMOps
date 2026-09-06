from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from qdrant_client.http.models import PointStruct, SparseVector
import uuid

# Global variable to store the single QdrantManager instance (Singleton Pattern)
_qdrant_instance = None

class QdrantManager:
    """
    Singleton Pattern: This ensures there is NEVER more than ONE open connection
    to the Qdrant database at the same time.
    Qdrant in local (file) mode only accepts a single client connection at a time.
    """

    def __new__(cls, *args, **kwargs):
        # __new__ is called BEFORE __init__ when creating an object.
        # We check if an instance already exists. If yes, we return it directly
        # without creating a new one.
        global _qdrant_instance
        if _qdrant_instance is None:
            _qdrant_instance = super().__new__(cls)
        return _qdrant_instance

    """
    This class handles all interactions with the Qdrant Vector Database.
    It creates the database, inserts our vectors (texts and images), 
    and executes "hybrid" searches.
    """
    def __init__(self, collection_name="multimodal_agent"):
        # The Singleton guarantees the same instance, but __init__ is still called every time.
        # We add a flag to ensure initialization only happens once.
        if getattr(self, '_initialized', False):
            return

        # Initialize Qdrant in "local" mode (storage on disk in the qdrant_data folder).
        # This avoids needing Docker or an external API. Infrastructure cost: 0 €.
        self.client = QdrantClient(path="./qdrant_data")
        self.collection_name = collection_name
        
        # Prepare the collection on startup.
        self._setup_collection()
        
        self._initialized = True

    def _setup_collection(self):
        """
        Checks if the collection exists. If not, creates it with 
        the necessary parameters for Hybrid Search (Dense + Sparse vectors).
        """
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating Qdrant collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=512, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )

    def recreate_collection(self):
        """
        Destroys the existing collection and creates a new, empty one.
        Used during a full re-ingestion (when the source files have changed).
        """
        print(f"Resetting Qdrant database '{self.collection_name}'...")
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        # Call _setup_collection to recreate a clean collection
        self._setup_collection()
        print("Qdrant database has been reset.")


    def insert_documents(self, documents: list[dict]):
        """
        Inserts our documents (transcribed texts or image slides) into Qdrant.
        
        Expected format of a document in the list:
        {
            "dense_vector": [0.1, 0.5, ...],
            "sparse_vector": {"indices": [...], "values": [...]},  # (Optional for images)
            "payload": {"text": "...", "start": 12.5, "type": "audio"} # Attached metadata
        }
        """
        points = []
        for doc in documents:
            # Each point in Qdrant needs a unique ID. We generate a random UUID.
            point_id = str(uuid.uuid4())
            
            # Prepare the vectors for insertion
            vector_dict = {"dense": doc["dense_vector"]}
            
            # If a sparse vector exists (often the case for text), add it.
            if "sparse_vector" in doc and doc["sparse_vector"]:
                vector_dict["sparse"] = SparseVector(
                    indices=doc["sparse_vector"]["indices"],
                    values=doc["sparse_vector"]["values"]
                )
                
            # Create the Qdrant "Point" which unites vectors and metadata (text, timestamp, etc.)
            points.append(PointStruct(
                id=point_id,
                vector=vector_dict,
                payload=doc["payload"]
            ))
            
        print(f"Inserting {len(points)} points into Qdrant...")
        # Insert everything at once in a batch (upsert = insert or update)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def hybrid_search(self, query_dense: list[float], query_sparse: dict = None, limit: int = 3):
        """
        Performs a "Hybrid" search leveraging Qdrant's capabilities.
        Qdrant will query both the Dense (meaning) and Sparse (keywords) layers simultaneously,
        and fuse the results to return the most relevant documents.
        """
        from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion
        
        # We will ask Qdrant to run multiple queries at the same time (Prefetching)
        prefetch = []
        
        # Dense Query (The "Meaning" / Semantic Search)
        prefetch.append(
            Prefetch(
                query=query_dense,
                using="dense",
                limit=limit
            )
        )
        
        # Sparse Query (The "Exact Keywords" / Lexical Search), if provided
        if query_sparse:
            prefetch.append(
                Prefetch(
                    query=SparseVector(
                        indices=query_sparse["indices"],
                        values=query_sparse["values"]
                    ),
                    using="sparse",
                    limit=limit
                )
            )
            
        # Run the search using Reciprocal Rank Fusion (RRF).
        # This is CRUCIAL so that images (which have a poor Dense semantic score compared to the text query)
        # still rank highly thanks to their extracted OCR text (which gives a very high Sparse score).
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True
        )
        
        # Return a clean list of payloads (the actual text and metadata)
        return [res.payload for res in results.points]
