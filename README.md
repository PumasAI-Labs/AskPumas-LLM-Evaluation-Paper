# Evaluation of Large Language Models for an AI Chat Assistant Focused on Pumas and Pharmacometrics

Running Python scripts to explore and compare different Large Language Model options requires downloading the required libraries using [`uv`](https://docs.astral.sh/uv/). To install `uv`, follow the [instructions on their documentation page](https://docs.astral.sh/uv/getting-started/installation/).
Questions asked to the Large Language Models are saved in `prompts.csv`.

Confidential data files
- `prompts.csv` is excluded from the repository for confidentiality. Use `prompts.csv.example` as a template to create your local `prompts.csv` from secure sources.
- `results.csv` is included in the repository (no example file is required); it stores aggregated LLM outputs or evaluation summaries produced by the pipeline.
- `multillm.csv` is the aggregated multi-LLM output produced by `main.py`. It contains a `Question` column plus one column per model (model names are defined in `model_options.json`) and is used by `embeddings-llm-output.py` to compute embeddings for each model's output and by downstream analysis.
- `ragas-results/*.csv` are per-model RAGAS evaluation CSVs (one file per evaluated LLM). These files contain rows with fields such as `user_input`, `retrieved_contexts`, `response`, `faithfulness`, and `answer_relevancy`, and store automated evaluation metrics for each prompt-model pair. Real `ragas-results/*.csv` files are excluded from the repository for confidentiality.

To run the scripts locally, create or place the following confidential files in the repository root (or `ragas-results/` folder as appropriate): `prompts.csv`, `multillm.csv`, and any `ragas-results/*.csv`. Use `prompts.csv.example` and `results.csv.example` as templates for `prompts.csv` and `results.csv` if helpful. `results.csv` and `multillm.csv` can be used as-is or replaced with your own files.


### Running Python Code

You must first install all dependencies and obtain appropriate API keys to run the code. 
All code is stored in the root directory. The following instructions assume that you are at the root of this repository.

- You will need to get an OpenRouter API key and store it in a `.env` file in the current working directory. To do this, setup an account through OpenRouter and generate an API key at https://openrouter.ai/settings/keys. Ensure to add enough credits to your OpenRouter account to run all the Large Language Model outputs.

All Python scripts should be run through `uv` (e.g. `uv run main.py`). 
If you haven't installed the dependencies, this will automatically download them. You could also run `uv sync` before running anything. 

- Run `main.py` to get an output of different questions asked to numerous Large Language Models. You can do this by calling the command `uv run main.py` in the terminal. This will create a CSV file of the captured output. In `main.py` you can explore other models by adding models to the `model_options` variable. Comment out models that you wish to skip when generating Large Language Model responses.

### Parsing Interview Questions

The `parse-interview-questions/` directory contains a script to parse interview transcripts and generate questions using GPT. To run it:

```bash
cd parse-interview-questions
uv run interview-parse-gpt.py
```

This requires an OpenAI API key set in your environment or `.env` file.

### Rendering the Research Paper

The research paper that evaluates Large Language Models (LLMs) for RAG applications is written in Quarto (`research-paper.qmd`) and can be rendered to a PDF using `uv` to manage the environment and dependencies.

To generate the PDF, run:

```bash
uv run quarto render research-paper.qmd
```
