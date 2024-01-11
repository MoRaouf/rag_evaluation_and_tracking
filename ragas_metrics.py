
import os
from ragas.metrics import (
  Faithfulness,
  AnswerRelevancy,
  ContextPrecision,
  ContextRecall,
  AnswerSimilarity
)
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_similarity
)

from ragas.metrics.critique import conciseness, harmfulness
from ragas import evaluate
# from ragas.embeddings import FastEmbedEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.callbacks.tracers import LangChainTracer
from ragas.llms import LangchainLLM

from langchain_openai import ChatOpenAI
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

# Define Embedding & LLM models
fast_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
gpt35_wrapper = LangchainLLM(llm=gpt35)

# LangSmith Tracer
tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT"))

# RAGAS Metrics 
# faithfulness = Faithfulness(
#     batch_size = 1,
#     llm = gpt35_wrapper
# )
faithfulness.llm = gpt35_wrapper

# answer_relevancy = AnswerRelevancy(
#   batch_size = 1,
#   llm=gpt35_wrapper,
#   embeddings=fast_embeddings
# )
answer_relevancy.llm=gpt35_wrapper
answer_relevancy.embeddings=fast_embeddings

# context_precision = ContextPrecision(
#   batch_size = 1,
#   llm=gpt35_wrapper
# )
context_precision.llm=gpt35_wrapper

# context_recall = ContextRecall(
#   batch_size = 1,
#   llm=gpt35_wrapper
# )
context_recall.llm=gpt35_wrapper

# answer_similarity = AnswerSimilarity(
#   batch_size = 1,
#   llm=gpt35_wrapper,
#   embeddings=fast_embeddings
# )
answer_similarity.llm=gpt35
answer_similarity.embeddings=fast_embeddings


def evaluate_ragas(ragas_dataset: Dataset):
  # print(type(ragas_dataset))
  result = evaluate(
    dataset=ragas_dataset,
    metrics=[
      answer_similarity,
      answer_relevancy,
      conciseness, 
      harmfulness,
      # faithfulness,
      # context_precision,
      # context_recall,
    ],
    # callbacks=[tracer],
  )
  return result