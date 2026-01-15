from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
import csv
import json
from pathlib import Path
from datasets import Dataset
from parseprompts import *
import sys

csv.field_size_limit(sys.maxsize)
load_dotenv()

# Constants
MODEL_OPTIONS_PATH = Path(__file__).parent / "model_options.json"
with open(MODEL_OPTIONS_PATH) as f:
    model_options = json.load(f)
MODEL_NAME = [model for models in model_options.values() for model in models]
FILE_NAME_ASKPUMAS_DATA = "prompts.csv"
FILE_NAME_LLM_OUTPUT = "multillm.csv"

# Functions
def get_prompts_context(input_data) -> dict:
    """
    Parses all question prompts and context data from 'prompts.csv' file.
    Returns a dictionary containing question prompts and their respective contexts.
    """
    data_samples = {"question": [], "contexts": []}
    for i in input_data:
        question_data = i["question"]
        context_data = i["question_context"]
        data_samples["question"].append(question_data)
        data_samples["contexts"].append([context_data])
    return data_samples


def get_llm_output(file_name, model_name) -> dict:
    """
    Parses the Large Language Model output generated from the 'multillm.csv' file.
    Returns a dictionary containing a list of the respective Large Language Model output.
    """
    data_samples = {"answer": []}
    with open(file_name, newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            if model_name in row:
                data_samples["answer"].append(row[model_name])
    return data_samples


def ragas_evalution_output(sampled_data_dict):
    """
    Automatically ecaluates Large Language Model outputs using the RAGAS Python Library.
    Compares model outputs with AskPumas, outputting a percentage score.
    Returns a Pandas dataframe, which includes columns of evaluation scores such as 'faithfulness' and 'ResponseRelevancy'
    """

    dataset = Dataset.from_dict(sampled_data_dict)
    score = evaluate(dataset, metrics=[Faithfulness(), ResponseRelevancy()])
    df = score.to_pandas()
    return df


# Gather data from respective functions
askpumas_prompts_context = parse_csv_prompts(FILE_NAME_ASKPUMAS_DATA)[1]
question_contexts_dict = get_prompts_context(askpumas_prompts_context)

# Evaluate outputs by iterating through all available Language Model and storing the results in a CSV file
for i in MODEL_NAME:
    answer_dict = get_llm_output(FILE_NAME_LLM_OUTPUT, i)
    sampled_data_dict = {**question_contexts_dict, **answer_dict}
    df_ragas_output = ragas_evalution_output(sampled_data_dict)
    model_name = i.replace("/", "_")
    df_ragas_output.to_csv("ragas_evaluation_" + f"{model_name}" + ".csv")
