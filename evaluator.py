import os
from datetime import datetime
from typing import Callable, Optional
import mlflow
import json
import pandas as pd
from ragas_metrics import evaluate_ragas

from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()


class Evaluator:
    def __init__(
            self,
            testset: Dataset,
            eval_func: Callable,
    ):

        self.testset = testset
        self.eval_func = eval_func

    def evaluate(
            self,
            exp_name: Optional[str] = None,
    ):
        
        # If no experiment name is provided, use the current timestamp
        if not exp_name:
            exp_name = "RAGAS_Evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a new experiment
        mlflow.set_experiment(exp_name)
        print("Started MLflow Experiment...")
        # Specify remote tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        # Start a run 
        with mlflow.start_run(run_name="run_" + datetime.now().strftime("%Y%m%d_%H%M%S")):
            # Log the used model
            logged_model = mlflow.openai.log_model(
                model="gpt-3.5-turbo-1106",
                task="chat.completions",
                artifact_path="eval_model",
                # input_example=self.testset["question"][0],
            )

            # Get RAGAS evaluation scores
            print("Started RAGAS Evaluation ...")
            result = evaluate_ragas(self.testset)
            print("Finished RAGAS Evaluation ...")
            # Save the results
            with open(f"scores/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as file:
                json.dump(result, file, indent=2)

            # Log scores result to MLflow
            mlflow.log_table(data=result.to_pandas(), artifact_file="scores_result.json")
            # Log testset to MLflow
            mlflow.log_table(data=self.testset.to_pandas(), artifact_file="testset.json")
