# rag_evaluation_and_tracking

## About 
This project houses a Retrieval Augmented Generation (RAG) LLM application built for robust and context-aware text generation. It leverages the combined power of LangChain for orchestration, MLflow for tracking and experimentation, DVC for version control, and RAGAS for evaluation.

## Technical Stack

- LangChain: Streamlines the data pipeline for retrieval and generation tasks.
- Qdrant: Vector Database to store embeddings of documents.
- MLflow: Manages experiments, tracks ML pipelines, and logs metrics.
- DVC: Facilitates version control and reproducibility of datasets and code.
- RAGAS: Offers comprehensive evaluation metrics for RAG systems.

## Evaluation Metrics

RAGAS empowers you to assess your RAG system's performance through various metrics. The ones used in this app are:

- **Answer Semantic Similarity**: Measures the meaning similarity between generated and ground-truth answers (0-1, higher is better).
- **Answer Relevance**: Evaluates how pertinent the answer is to the prompt (0-1, higher is better).
- **Answer Correctness**: Assesses the factual accuracy of the generated answer (0-1, higher is better).
- **Harmfulness**: Detects harmful language and information in the output.