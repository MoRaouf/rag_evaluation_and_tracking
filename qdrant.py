import os

import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import UpdateStatus, Filter
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import FastEmbedEmbeddings

# from fastembed.embedding import DefaultEmbedding

from typing import Iterable, Optional, List, Sequence



#===================================================================================
#---------------------------------- Qdrant Class  ----------------------------------
#===================================================================================


class QdrantVectorStore():
    """Class for Qdrant Vector Store"""

    def __init__(self,
                 local: str = True,
                 url: str = None,
                 api_key: str = None,
                 ):
        
        """Instantiate a Client instance"""
        if local:
            os.makedirs("/qdrant_db", exist_ok=True)
            self.client = qdrant_client.QdrantClient(path="/qdrant_db")
        else:
            assert url and api_key, "Please add your cloud instance URL & API_Key"
            self.client = qdrant_client.QdrantClient(
                url = url,
                api_key = api_key
                )
            
        self.default_embeddings_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
    #===================================================================================
    
    def create_collection(self,
                          collection_name: str,
                          documents: List[Document], # ADDED
                          embeddings_size: int = None,
                          embeddings_model: Embeddings = None, # ADDED
                          hybrid_embeddings: bool = False,
                          ):
        """Create new collection in Qdrant with Index"""

        if embeddings_size is None:
            embeddings_size = self.default_embeddings_model.max_length

        # Check if collection exists, then prompt user to query the collection or choose another name
        # try:
        #     collection_info = self.client.get_collection(collection_name=collection_name)
        #     # if collection_info.status == CollectionStatus.GREEN:
        #     if collection_info:
        #         raise Exception((f"Collection {collection_name} already exists, "
        #                         "please specify anothor name for the new collection, "
        #                         "or query the existing collection using `query_collection`"))
        # except Exception as e:
        #     pass

        #### ADDED ---------------------------------

        # Use FastEmbed Default Embedding Model "BAAI/bge-small-en-v1.5"
        if embeddings_model is None:
            embeddings_model = self.default_embeddings_model

        # Qdrant instance from QdrantClient
        qdrant = Qdrant.from_documents(
            documents,
            embeddings_model,
            path="qdrant_db/",
            collection_name="demo_collection",
        )
        #### ADDED ---------------------------------

        # if hybrid_embeddings:
        #     # Create new collection with dense & sparse vectors
        #     self.client.create_collection(
        #         collection_name=collection_name,
        #         vectors_config={
        #             "dense_vectors": models.VectorParams(
        #                 size=embeddings_size, 
        #                 distance=models.Distance.COSINE,
        #                 ),
        #         },
        #         sparse_vectors_config={
        #             "sparse_vectors": models.SparseVectorParams(),
        #         }, 
        #     )
        # else:
        #     # Create new collection with dense vectors only
        #     self.client.create_collection(
        #         collection_name=collection_name,
        #         vectors_config={
        #             "dense_vectors": models.VectorParams(
        #                 size=embeddings_size, 
        #                 distance=models.Distance.COSINE
        #                 ),
        #         }
        #     )
    
    #===================================================================================

    def list_collections(self):
        # List all existing collections

        return self.client.get_collections()

    #===================================================================================

    def add_texts(self,
                collection_name: str,
                texts: Iterable[str],
                embeddings_model: Embeddings = None,
                metadatas: Optional[List[dict]] = None,
                ids:Optional[Sequence[str]] = None,
                ):
        """Add new text data to an existing collection in Qdrant"""

        # Use FastEmbed Default Embedding Model "BAAI/bge-small-en-v1.5"
        if embeddings_model is None:
            embeddings_model = self.default_embeddings_model

        # Qdrant instance from QdrantClient
        qdrant = Qdrant(
            client=self.client, 
            collection_name=collection_name, 
            embeddings=embeddings_model
        )

        added_payloads = qdrant.add_texts(
            texts=texts,
            metadatas=metadatas if metadatas is not None else [],
            ids=ids if ids is not None else [],
        )

        print("Added Text Payloads -> Completed")
        # return ("Added Text Payloads Status is COMPLETED: ", added_payloads.status == UpdateStatus.COMPLETED)
    
    #===================================================================================

    def add_documents(self,
                collection_name: str,
                documents: List[Document],
                embeddings_model: Embeddings = None,
                ):
        """Add new documents to an existing collection in Qdrant"""

        # Use FastEmbed Default Embedding Model "BAAI/bge-small-en-v1.5"
        if embeddings_model is None:
            embeddings_model = self.default_embeddings_model

        # Qdrant instance from QdrantClient
        qdrant = Qdrant(
            client=self.client, 
            collection_name=collection_name, 
            embeddings=embeddings_model
        )

        added_payloads = qdrant.add_documents(
            documents=documents,
        )
        
        print("Added Document Payloads -> Completed")
        # return ("Added Document Payloads Status is COMPLETED: ", added_payloads.status == UpdateStatus.COMPLETED)
    
    #===================================================================================

    def similarity_search(self,
                          input_query:str,
                          collection_name: str,
                          embeddings_model: Embeddings = None,
                          filter: Optional[Filter] = None,
                          score_threshold: Optional[float] = None,
                          top_k=10,
                          ):
        """Return docs most similar to query.

        Args:
            collection_name (str): collection name to search
            embeddings_model (Embeddings): embeddings model to embed input query
            input_query (str): Text to look up documents similar to
            filter (Optional[Filter], optional): Filter by metadata. Defaults to None.
            score_threshold (Optional[float], optional): Define a minimal score threshold for the result.
                If defined, less similar results will not be returned. Defaults to None.
            top_k (int, optional): Number of Documents to return. Defaults to 10.

        Returns:
            List[Document]: List of Documents most similar to the query.
        """

        # Use FastEmbed Default Embedding Model "BAAI/bge-small-en-v1.5"
        if embeddings_model is None:
            embeddings_model = self.default_embeddings_model

        # Qdrant instance from QdrantClient
        qdrant = Qdrant(
            client=self.client, 
            collection_name=collection_name, 
            embeddings=embeddings_model
        )
        
        query_results = qdrant.similarity_search(
            query=input_query,
            k=top_k,
            filter=filter,
            score_threshold= score_threshold,
        )
        
        return query_results
        
    #===================================================================================

    def delete_collection(self,
                          collection_name: str,
                          ):
        """Delete collection from Qdrant"""

        self.client.delete_collection(collection_name=collection_name)

    #===================================================================================

    def compute_sparse_vector(input_text: str):
        # Create the sparse vector embedding using BM25
        pass 

    #===================================================================================

    def qdrant_retriever(self,
                         collection_name: str,
                         embeddings_model: Embeddings = None,
                         ):
        """Return Qdrant as a retriever for LangChain

        Args:
            collection_name (str): The collection to search
            embeddings_model (Embeddings): Embeddings model to embed search queries

        Returns:
            Retrieever (VectorStoreRetriever): A qdrant retriever instance
        """

        # Use FastEmbed Default Embedding Model "BAAI/bge-small-en-v1.5"
        if embeddings_model is None:
            embeddings_model = self.default_embeddings_model

        # Qdrant instance from QdrantClient
        qdrant = Qdrant(
            client=self.client, 
            collection_name=collection_name, 
            embeddings=embeddings_model
        )
        
        return qdrant.as_retriever()