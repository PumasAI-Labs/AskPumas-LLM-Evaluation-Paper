import csv, sys, os

file_name = "prompts.csv"


def parse_csv_prompts(input_file):
    with open(input_file, "r") as file:
        csv_reader = csv.reader(file)
        headers_list = next(csv_reader)
        prompt_list_final = []
        prompt_list_final = [
            {headers_list[i]: row[i] for i in range(len(headers_list))}
            for row in csv_reader
        ]
    return [headers_list, prompt_list_final]


output = parse_csv_prompts(file_name)
