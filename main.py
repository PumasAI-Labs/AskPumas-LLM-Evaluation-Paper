from parseprompts import *
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
from prompt_loader import load_prompt
import asyncio

load_dotenv()
if not os.getenv("OPENROUTER_API_KEY"):
    raise Exception(
        "OpenRouter API Key is not set. Please edit the OPENROUTER_API_KEY variable in your '.env' file"
    )


FILE_NAME = "prompts.csv"
OVERALL_TEMPERATURE = 0
MODEL_OPTIONS_PATH = Path(__file__).parent / "model_options.json"
with open(MODEL_OPTIONS_PATH) as f:
    model_options = json.load(f)


async def call_local_llm(name, provider, data, system_prompt):
    """
    Calls a specified LLM with a given prompt and returns the output.

    - Extracts the question and question context from the provided data.
    - Constructs a conversation with system prompt and user query.
    - Invoke the language model with the conversation messages.
    - Returns the output text from the language model along with the original question in a tuple.
    """
    question = data["question"]
    user_prompt = data["question_context"]
    chain = ChatOpenAI(
        model=name,
        temperature=OVERALL_TEMPERATURE,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    conversation = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = await chain.ainvoke(conversation)
    return response.content, question


async def run_all_llm(model_option_dict, question_data):
    """
    Runs all specified language models on a given question and returns their responses.

    - Loads the system prompt from template.
    - Iterates through each language model from the model_options dictionary and calls the model using the call_local_llm() function with the question data.
    - Returns the responses from all models along with the original question in a dictionary.
    """
    system_prompt = load_prompt("system_prompt")

    tasks = []
    model_names = []
    for model_provider, model_list in model_option_dict.items():
        for model_name in model_list:
            tasks.append(
                call_local_llm(model_name, model_provider, question_data, system_prompt)
            )
            model_names.append(model_name)

    results = await asyncio.gather(*tasks)

    final_output_dict = {}
    for model_name, (captured_output_from_llm, question_sent) in zip(
        model_names, results
    ):
        final_output_dict[model_name] = captured_output_from_llm
    final_output_dict["Question"] = question_sent

    return final_output_dict


def store_data_json(
    final_output_list,
):
    """
    Stores the results of multiple language models in a CSV file.

    - Defines the headers for the CSV file, including the question and each model's name.
    - Opens a CSV file for writing and initializes a DictWriter with the defined headers.
    - Writes each row of results from the final output list to the CSV file labled "multillm.csv".
    """
    headers = ["Question"]
    for model_provider, model_list in model_options.items():
        for model_name in model_list:
            headers.append(model_name)
    with open("multillm.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(final_output_list)


async def run(input_all_models):
    """
    Executes the LLM generation for all prompts from "prompts.csv".

    - Parses the CSV file containing prompts and their context using the parse_csv_prompts() function.
    - Iterates through each prompt, runs all specified models, and collects the responses.
    - Stores the collected responses in a CSV file using the store_data_json() function.
    """
    csv_data = parse_csv_prompts(FILE_NAME)[1]
    final_output_list = []
    for i in tqdm(csv_data):
        data_input = i
        result = await run_all_llm(input_all_models, data_input)
        final_output_list.append(result)
        await asyncio.sleep(1)
    store_data_json(final_output_list)


asyncio.run(run(model_options))
