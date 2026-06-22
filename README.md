# Harkalm Property Listing Classifier

A Python-based AI tool designed to parse, preprocess, and classify commercial property listings into business sectors relevant to the Harkalm Group's acquisition strategy:
- **Nursery**
- **SEN School**
- **Food Store**
- **None** (unmatched or ambiguous properties)

The classifier uses OpenAI's structured outputs (`gpt-4o-mini`) via a Pydantic validation schema, ensuring reliable, parseable results without forcing incorrect classifications on ambiguous or weak data.

## Project Structure
* `main.py` - Orchestrator that parses CLI arguments, runs the ingestion, invokes classification, and writes results.
* `classifier.py` - Logic for text aggregation, LLM schema integration, API retry management, and Mock Mode validation.
* `prompt.py` - Definition of the system prompts and instruction patterns.
* `utils.py` - Helper utilities including logging setup and safe string-list parsers.
* `requirements.txt` - Project dependencies.
* `.env.example` - Example configuration file for environment variables.
* `submission_notes.md` - Analysis of assumptions, ambiguity strategy, and areas of future improvement.

## Mock Mode
To make local evaluation as easy as possible, the script has a built-in **Mock Mode**.
- If `OPENAI_API_KEY` is not found or is empty in your `.env` file, the script will display a warning and run using local, deterministic mock classifications.
- The mock classifications use simple keyword-based heuristics (analyzing text tokens for target sectors) to simulate classification outcomes locally.
- Once you provide a valid API key, the script will automatically switch to making live OpenAI API calls.

## Setup Instructions

1. Clone or extract this repository.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
5. Place the input file `listings.csv` in the root directory.

## Usage

To run the classifier on the full dataset:
```bash
python main.py
```

Options:
- `--dry-run`: Verify ingestion, keyFeatures parsing, and aggregation without calling the LLM or generating output.
- `--limit N`: Limit classification to the first N listings (ideal for testing/cost control).
