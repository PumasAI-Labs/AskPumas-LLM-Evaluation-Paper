from spire.doc import *
from spire.doc.common import *
from dotenv import load_dotenv
from openai import OpenAI
import time
from tqdm import tqdm
import csv
import os

load_dotenv()
MODEL_EVAL_NAME = "gpt-4o"
OVERALL_TEMP = 0
CURRENT_PATH = os.path.join(os.path.dirname(__file__))
FILE_OUTPUT_NAME = f"interview_questions_{MODEL_EVAL_NAME}_temp_{OVERALL_TEMP}.csv"
FILE_PATH = os.path.join(os.path.join(CURRENT_PATH, "gptoutputs"), FILE_OUTPUT_NAME)
client = OpenAI()


document_list_names = os.listdir(CURRENT_PATH)
document_list_names = [name for name in document_list_names if "docx" in name]


def parse_doc(document_name: str) -> str:
    """
    Parses an inteview transcript document in ".docx" format and extract its text content into string.
    Returns a string - the extracted text from the document
    """
    document = Document()
    document.LoadFromFile(document_name)
    output_string = document.GetText()
    return output_string


def call_gpt_api(input_interview_str: str) -> str:
    """
    Calls the OpenAI GPT API to generate questions based on input interview transcript.
    Takes in the inputted string and calls "gpt-4-turbo"
    Returns generated questions in the form of a string from the model.
    """
    prompt = """
                Task Description:
                    * You will receive an interview transcript between two interviewers and one consultant. In this interview, the consultant discusses Pumas, a package in the Julia Programming Language.
                    * Your goal is to generate technical questions that could be posed to an AI Chatbot, based solely on the consultant's answers about Pumas. You must ignore the interviewers' questions.
                Objective:
                    * From the consultant's discussion on Pumas, infer relevant details and formulate questions. These prompts will later test the AI Chatbot's understanding and responsiveness.
                About Pumas:
                    * Pumas (PharmaceUtical Modeling And Simulation) is a suite of tools within the Julia Programming Language. It facilitates quantitative analytics across pharmaceutical drug development phases.
                    * The framework aims to streamline various aspects of analytics in drug development, providing efficient implementations under one cohesive package.
                    * There are several related packages and tools that complement Pumas functionalities, enhancing its utility in pharmaceutical modeling and simulation.
                Instructions:
                    * Analyze the interview transcript focusing only on the consultant's discussions related to Pumas.
                    * Based on their insights and explanations, generate thoughtful technical questions that could be posed to an AI Chatbot.
                    * Ensure the questions cover various aspects of Pumas to comprehensively evaluate the Chatbot's capabilities.

             """
    completion = client.chat.completions.create(
        model=MODEL_EVAL_NAME,
        temperature=OVERALL_TEMP,
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"""{input_interview_str}""",},
        ],
    )
    response = completion.choices[0].message.content
    return response


def run(document_list_names_interviews: list) -> list:
    """
    Processes a list of interview transcript documents.
    Then, extracts the text using the parse_doc() function and generate questions using GPT with the call_gpt_api() function.
    Creates a list and stores the list in a CSV file to read 
    """
    list_of_potential_questions = []
    for document_string_name in tqdm(
        document_list_names_interviews, desc="Processing documents"
    ):
        gpt_output = call_gpt_api(parse_doc(document_string_name))
        list_of_potential_questions.append(gpt_output)
        time.sleep(0.1)
    with open(FILE_PATH, "w", newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["Document Name", "Question"])
        for document_name, question in zip(
            document_list_names_interviews, list_of_potential_questions
        ):
            csvwriter.writerow([document_name, question])


run(document_list_names)
