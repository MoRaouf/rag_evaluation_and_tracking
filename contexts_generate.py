from qdrant import QdrantVectorStore

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant

import qdrant_client
import json

# ------------------------ Vector Store ------------------------
# Qdrant Client
# client = QdrantVectorStore()
# # client.delete_collection("demo_collection")
# retriever = client.qdrant_retriever(
#     collection_name="new_collection"
# )

client = qdrant_client.QdrantClient(path="qdrant_db")
embeddings_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
qdrant = Qdrant(client, "new_collection", embeddings_model)

# ------------------------ Manual Testset Generation ------------------------
# Get a group of documents to generate questions & answers for the manual testset
manual_retriever_query = "What are the main principles mentioned?"
# docs_for_manual_testset = retriever.get_relevant_documents(query=manual_retriever_query, k=5)
docs_for_manual_testset = qdrant.similarity_search(query=manual_retriever_query, k=5)

txt_docs_for_manual_testset = [doc.page_content for doc in docs_for_manual_testset]

# with open("data/manual_docs.txt", 'w') as file:
#     for item in txt_docs_for_manual_testset:
#         file.write(item + "\n\n\n"+ "-"*20 + "\n")

with open("data/manual_docs.json", 'w') as file:
    json.dump(txt_docs_for_manual_testset, file, indent=2)

# ------------------------ RAGAS Synthetic Testset Generation ------------------------
# Get a group of documents to generate RAGAS testset
ragas_retriever_query = "How to implement & fix the issues?"
# docs_for_ragas_testset = retriever.get_relevant_documents(query=ragas_retriever_query, k=5)
docs_for_ragas_testset = qdrant.similarity_search(query=ragas_retriever_query, k=5)

txt_docs_for_ragas_testset = [doc.page_content for doc in docs_for_ragas_testset]

# with open("data/ragas_docs.txt", 'w') as file:
#     for item in txt_docs_for_ragas_testset:
#         file.write(item + "\n\n\n"+ "-"*20 + "\n")

with open("data/ragas_docs.json", 'w') as file:
    json.dump(txt_docs_for_ragas_testset, file, indent=2)