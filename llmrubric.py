from openai import OpenAI
import pandas as pd
import json, os, csv
from pathlib import Path
from dotenv import load_dotenv
import sys

csv.field_size_limit(sys.maxsize)
load_dotenv()

# Constants
MODEL_OPTIONS_PATH = Path(__file__).parent / "model_options.json"
with open(MODEL_OPTIONS_PATH) as f:
    model_options = json.load(f)
MODELS = [model for models in model_options.values() for model in models]
RUBRIC_FILE_NAME = "llmrubric.csv"
MULTILLM_FILE_NAME = "multillm.csv"
LLM_RUBRIC_OUTPUT_FILE = "LLM-rubric-evaluation.csv"
MODEL_EVAL_NAME = "gpt-5-mini"
client = OpenAI()

# Functions
def parse_csv_data(file) -> str:
    """
    Parses rubric data from 'llmrubric.csv' file. 
    This data will be used by the Large Language Model to evaluate given outputs.
    Returns a string of data, formatted as a list of dictionaries. This is so that the Large Language Model can easily read the formatted data.
    """
    with open(file) as csvFile:
        csv_reader = csv.DictReader(csvFile)
        data = [row for row in csv_reader]
        data_str = str(data)
        return data_str


def get_llm_output(file_name, model_name) -> dict:
    """
    Parses the Large Language Model output generated from the 'multillm.csv' file.
    Returns a dictionary containing a list of the respective Large Language Model outputs and a list of their respective question prompts.
    """
    data_samples = {}
    data_samples["answer"] = []
    data_samples["question"] = []
    with open(file_name, newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            data_samples["question"].append(row["Question"])
            if (
                model_name in row
            ):  # Checking specifically for the Langauge Model that is being evaluated
                data_samples["answer"].append(row[model_name])
    return data_samples


def create_dataframe(model_data):
    """
    Parses evaluated data into a dictionary, and then into a Pandas DataFrame.
    This is to succesfuly store the evaluated data into a CSV file.
    Returns a Pandas DataFrame.
    """
    data = []
    for model_name, questions in model_data.items():
        for i, scores in enumerate(questions):
            row = {
                "Question": f"Question {i + 1}",
                "Model Name": model_name,
                "Relevance": scores["Relevance"],
                "Comprehensiveness": scores["Comprehensiveness"],
                "Clarity and Coherence": scores["Clarity and Coherence"],
                "Depth and Detail": scores["Depth and Detail"],
                "Total Score": scores["Total"],
            }
            data.append(row)
    df = pd.DataFrame(data)
    return df


def run(rubric, question, answer):
    """
    Evaluate Large Langauge Model output answers by taking the question and evaluation rubric as context.
    Uses prompt engineering to obtain a Python JSON object by making API calls to OpenAI's 'gpt-5-mini'.
    Returns a Python JSON object of the GPT evaluation, based on specific crtiera given to the model.
    """
    prompt = f"""   
        You will be given a question, answer, and rubric. 
        Use the rubric to grade the answer, based on the question being asked.
        Your response must be in a JSON format where you output a score based on the given rubric critera.
        For example, a valid output would contain each rubric critera, with a score like:
                "Relevance": 15,
                "Comprehensiveness": 25,
                "Clarity and Coherence": 20,
                "Depth and Detail": 30
    """
    completion = client.chat.completions.create(
        model=MODEL_EVAL_NAME,
        temperature=1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {
                "role": "user",
                "content": f"""
        Question: {question}
        Answer: {answer}
        Rubric: {rubric}
             """,
            },
        ],
    )
    response = completion.choices[0].message.content
    return response


# Evaluation of outputs
rubric_data = parse_csv_data(RUBRIC_FILE_NAME)
final_output_eval = {}
# Iterating through all avaialble models
for i in MODELS:
    evaluation_model = []
    model_output_dict = get_llm_output(MULTILLM_FILE_NAME, i)
    question_list = model_output_dict["question"]
    answer_list = model_output_dict["answer"]
    for question_str, answer_str in zip(question_list, answer_list):
        resp_out = run(rubric_data, question_str, answer_str)
        resp_out_dict = json.loads(
            resp_out
        )  # Convert JSON format back to a python object
        total_sum = sum(
            resp_out_dict.values()
        )  # Calculate the sum of the scores to store it as a new key-value pair in a dictionary
        resp_out_dict["Total"] = total_sum
        evaluation_model.append(resp_out_dict)
    final_output_eval[i] = evaluation_model
# Parsing and storing data appropriately
df = create_dataframe(final_output_eval)
df.to_csv(LLM_RUBRIC_OUTPUT_FILE, index=False)
