# RAG app evaluation with MLflow & RAGAS
import os
import requests
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from datasets import Dataset

import qdrant_client
from qdrant import QdrantVectorStore
from testset import generate_ragas_synthetic_testset, generate_testset_from_contexts, add_ground_truths, stack_datasets
from evaluator import Evaluator
from ragas_metrics import evaluate_ragas
import json

from dotenv import load_dotenv
load_dotenv()

# ------------------------ Data ------------------------
# url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
# res = requests.get(url)
# with open("data/state_of_the_union.txt", "w") as f:
#     f.write(res.text)

# # Load the data
# loader = TextLoader('data/state_of_the_union.txt')
# documents = loader.load()

# # Chunk the data
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = text_splitter.split_documents(documents)

# # ------------------------ Vector Store ------------------------

client = qdrant_client.QdrantClient(path="qdrant_db")
embeddings_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
qdrant = Qdrant(client, "new_collection", embeddings_model)

# added_docs = qdrant.add_documents(chunks)
# print(added_docs)


# Qdrant Client
# client = QdrantVectorStore()

# try:
#     client.create_collection(collection_name="demo_collection")
# except:
#     pass
# DONE -- Embedd chunks & add to Qdrant Vector Store -- DONE
# client.add_documents(
#     collection_name="demo_collection",
#     documents=chunks,
# )

retriever = qdrant.as_retriever()

# retriever = client.qdrant_retriever(
#     collection_name="new_collection"
# )

# Define LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)

# ------------------------ Question Generation Chain ------------------------
question_template = """\
You are a University Professor creating a test for advanced students. \
For each given context, create a question that is specific to the context. Avoid creating generic or general questions.\

### context: 
{context}

### Question:
"""
# Create Question Generation Chain
question_prompt = ChatPromptTemplate.from_template(question_template)
output_parser = StrOutputParser()
question_generation_chain = {"context": RunnablePassthrough()} | question_prompt | llm | output_parser

# ------------------------ Answer Generation Chain ------------------------
answer_template = """\
You are a University Professor creating a test for advanced students. For each given question and context, create an answer.\

### Question: 
{question} 

###Context: 
{context} 

Answer:
"""

answer_prompt = ChatPromptTemplate.from_template(answer_template)

# Setup RAG pipeline
answer_generation_chain = (
    {"context": RunnablePassthrough(),  "question": RunnablePassthrough()} 
    | answer_prompt 
    | llm
    | StrOutputParser() 
)

# # ------------------------ Manual Testset Generation ------------------------
# # Get a group of documents to generate questions & answers for the manual testset
# manual_retriever_query = "What are the main principles mentioned?"
# docs_for_manual_testset = retriever.get_relevant_documents(manual_retriever_query, k=5)

# # Generate questions from contexts
# manual_testset = generate_testset_from_contexts(
#     contexts=docs_for_manual_testset,
#     question_generation_chain=question_generation_chain,
#     answer_generation_chain=answer_generation_chain,
#     retriever=retriever,
# )

# ------------------------ RAGAS Synthetic Testset Generation ------------------------
# Get a group of documents to generate RAGAS testset
# ragas_retriever_query = "How to implement & fix the issues?"
# # docs_for_ragas_testset = retriever.get_relevant_documents(ragas_retriever_query, k=5)
# docs_for_ragas_testset = qdrant.similarity_search(query=ragas_retriever_query, k=5)
# # print(docs_for_ragas_testset, "\n\n", len(docs_for_ragas_testset))
# # Generate synthetic testset
# ragas_testset = generate_ragas_synthetic_testset(
#     documents=docs_for_ragas_testset,
# )

# ------------------------ Full Testset ------------------------
# Add Ground Truths to both generated testset
# Manual Testset
with open("data/manual_ground_truth.json", 'r') as file:
    manual_ground_truths = json.load(file)

with open("data/manual_testset.json", 'r') as file:
    manual_testset = json.load(file)

full_contexts = []
for i, q in enumerate(manual_testset["question"]):
    relevant_docs = retriever.get_relevant_documents(q, k=5)
    full_contexts.append([doc.page_content for doc in relevant_docs])
manual_testset["contexts"] = full_contexts

with open("data/manual_testset.json", 'w') as file:
    json.dump(manual_testset, file, indent=2)

full_manual_testset = add_ground_truths(
    dataset=Dataset.from_dict(manual_testset),
    ground_truths=manual_ground_truths
)

with open("data/full_manual_testset.json", 'w') as file:
    json.dump(full_manual_testset.to_dict(), file, indent=2)


# RAGAS Testset
# with open("data/ragas_docs.json", 'r') as file:
#     contexts_ragas = json.load(file)

# ground_truths_ragas = []
# add_ground_truths(
#     dataset=ragas_testset,
#     ground_truths=ground_truths_ragas
# )

# # Combine synthetic testset with the manual testset
# stacked_testset = stack_datasets([manual_testset, ragas_testset])
# print(type(full_manual_testset))
# print(type(full_manual_testset["contexts"]))
# print(full_manual_testset.features["contexts"])
    
# Evaluate testset
evaluator = Evaluator(
    testset=full_manual_testset,
    eval_func=evaluate_ragas
)

evaluator.evaluate(exp_name="RAGAS_Eval")
print("Finished .............................................")