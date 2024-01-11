from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document
from langchain_core.runnables import Runnable
from ragas.llms import LangchainLLM
from langchain_community.embeddings import FastEmbedEmbeddings
from ragas.testset import TestsetGenerator

from datasets import Dataset, concatenate_datasets
from typing import List, Optional
import json
from dotenv import load_dotenv
load_dotenv()

def generate_ragas_synthetic_testset(
        documents: List[Document],
        llm: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo-1106"),
        embeddings_model: FastEmbedEmbeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
) -> Dataset:
    
    # Define Embedding & LLM models
    generator_llm = LangchainLLM(llm=llm)
    critic_llm = LangchainLLM(llm=llm)
    fast_embeddings = embeddings_model

    # Change resulting question type distribution
    testset_distribution = {
        "simple": 0.25,
        "reasoning": 0.5,
        "multi_context": 0.0,
        "conditional": 0.25,
    }

    # Percentage of conversational question
    qa_percent = 0.2

    # Create instance of TestsetGenerator
    test_generator = TestsetGenerator(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings_model=fast_embeddings,
        testset_distribution=testset_distribution,
        chat_qa=qa_percent,
    )

    print("Started generating RAGAS Testset ....")
    testset = test_generator.generate(documents, test_size=5)
    print("Finished generating RAGAS Testset. _/")
    # Save test set
    testset.to_pandas().to_csv("data/testset.csv")
    
    return Dataset.from_dict(testset.to_pandas().to_dict('list'))


def add_ground_truths(
        dataset: Dataset, 
        ground_truths: List[str],
) -> Dataset:
    """Add ground truths to the synthetically generated testset

    Args:
        dataset (Dataset): The synthetically generated testset by RAGAS
        ground_truths (List[str]): List of ground truths to every generated question

    Returns:
        Dataset: A complete dataset for RAGAS evaluation
    """
    pd_dataset = dataset.to_pandas()
    # pd_dataset["contexts"] = pd_dataset["contexts"].map(lambda x: list(x)) 
    pd_dataset["ground_truths"] = [[gt] for gt in ground_truths]

    return Dataset.from_pandas(pd_dataset)


def generate_testset_from_questions_groundtruths(
        questions: List[str], 
        ground_truths: List[str], 
        rag_chain, 
        retriever,
) -> Dataset:
    """Manually generate testset for RAGAS evaluation

    Args:
        questions (List[str]): List of input questions to query RAG pipeline & get answers for.
        ground_truths (List[str]): List of reference answers for input questions
        rag_chain (_type_): RAG chain that will generate answers for input questions
        retriever (_type_): Vector Store Retriever that will retrieve relevant documents

    Returns:
        Dataset: A complete dataset for RAGAS evaluation
    """

    answers = []
    contexts = []

    # Generate answers & retrieve relevant documents
    for query in questions:
        answers.append(rag_chain.invoke(query))
        contexts.append([doc.page_content for doc in retriever.get_relevant_documents(query)])

    # To dict
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    return dataset


def generate_testset_from_contexts(
        contexts: Document,
        question_generation_chain: Runnable,
        answer_generation_chain: Runnable, 
        retriever,
) -> Dataset:
    "Generate testset by using contexts & LLM to generate questions & answers."

    print("Started generating Manual Testset ....")
    qac = {} # Question-Answer-Context
    qac["question"] = list()
    qac["answer"] = list()
    qac["contexts"] = list()
    for doc in contexts:
        generated_question = question_generation_chain.invoke({"context" : doc.page_content})
        qac["question"].append(generated_question)
        qac["contexts"].append(doc.page_content)

    for question, context in zip(qac["question"], qac["contexts"]):
        generated_answer = answer_generation_chain.invoke({"question": question, "context": context})
        qac["answer"].append(generated_answer)

    #TO DO
    #Retrieve relevant documents for every generated question
        

    with open("data/manual_testset.json", 'w') as file:
        json.dump(qac, file, indent=2)

    print("Finished generating Manual Testset. _/")
    return Dataset.from_dict(qac)

def stack_datasets(
        dataset_a,
        dataset_b,
):
    "Stack two datasets together."
     
    return concatenate_datasets([dataset_a, dataset_b])