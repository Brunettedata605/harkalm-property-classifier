# Harkalm Property Listing Classifier

A Python-based AI tool designed to parse, preprocess, and classify commercial property listings into business sectors relevant to the Harkalm Group's acquisition strategy:
- **Nursery**
- **SEN School**
- **Food Store**
- **None** (unmatched or ambiguous properties)

The classifier leverages OpenAI's structured outputs (`gpt-4o-mini`) via a Pydantic validation schema, ensuring reliable, parseable results without forcing incorrect classifications on ambiguous or weak data.

## Project Structure
* `main.py` - Orchestrator that parses CLI arguments, runs the ingestion, invokes classification, and writes results.
* `classifier.py` - Logic for text aggregation, LLM schema integration, and API retry management.
* `prompt.py` - Definition of the system prompts and instruction patterns.
* `utils.py` - Helper utilities including robust logging setup and safe string-list parsers.
* `requirements.txt` - Project dependencies.
* `.env.example` - Example configuration file for environment variables.
* `submission_notes.md` - Analysis of assumptions, ambiguity strategy, and areas of future improvement.

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
4. Copy `.env.example` to `.env` and fill in your OpenAI API Key:
   ```bash
   copy .env.example .env
   ```
5. Place the input file `listings.csv` in the root directory.

## Usage

To run the classifier on the dataset:
```bash
python main.py
```

Options:
- `--dry-run`: Verify ingestion and preprocessing steps without calling the OpenAI API.
- `--limit N`: Only run classification on the first N rows (recommended for cost-saving/testing).
