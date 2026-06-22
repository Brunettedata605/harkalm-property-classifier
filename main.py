import argparse
import logging
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Ensure the local path is in import search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logging
from classifier import preprocess_row, PropertyClassifier
from prompt import USER_PROMPT_TEMPLATE

# Load environment variables
load_dotenv()

def main():
    setup_logging()
    logger = logging.getLogger("main")
    
    parser = argparse.ArgumentParser(description="Harkalm Group Property Listing Classifier")
    parser.add_argument("--input", default="listings.csv", help="Path to input CSV file")
    parser.add_argument("--output", default="classified_listings.csv", help="Path to output CSV file")
    parser.add_argument("--dry-run", action="store_true", help="Preprocess listings and print details without calling the LLM")
    parser.add_argument("--limit", type=int, help="Limit classification to the first N listings")
    
    args = parser.parse_args()
    
    # 1. Verify input file existence
    if not os.path.exists(args.input):
        logger.error(f"Input file not found at {args.input}")
        sys.exit(1)
        
    logger.info(f"Loading data from {args.input}")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        sys.exit(1)
        
    logger.info(f"Loaded {len(df)} listings.")
    
    # Apply limit if specified
    if args.limit is not None:
        logger.info(f"Limiting processing to the first {args.limit} rows.")
        df_to_process = df.head(args.limit)
    else:
        df_to_process = df
        
    # 2. Preprocess data and run dry-run validation
    preprocessed_records = []
    logger.info("Preprocessing listings data...")
    for idx, (_, row) in enumerate(df_to_process.iterrows()):
        payload = preprocess_row(row)
        preprocessed_records.append((idx, payload))
        
    if args.dry_run:
        logger.info("--- DRY RUN MODE: Preprocessed Listings Preview ---")
        for idx, payload in preprocessed_records:
            user_prompt = USER_PROMPT_TEMPLATE.format(
                id=payload["id"],
                address=payload["address"],
                property_subtype=payload["property_subtype"],
                use_class=payload["use_class"],
                summary=payload["summary"],
                key_features=payload["key_features"],
                points_of_interest=payload["points_of_interest"],
                description=payload["description"]
            )
            print(f"\n=========================================")
            print(f"Record #{idx+1} | ID: {payload['id']}")
            print(f"=========================================")
            print(user_prompt)
        logger.info("Dry-run preview complete. No API calls were made.")
        return
        
    # 3. Running actual classification (Milestone 3 skeleton)
    logger.info("Initializing Property Classifier...")
    classifier = PropertyClassifier()
    
    categories = []
    confidences = []
    reasonings = []
    
    logger.info("Classifying listings...")
    for idx, payload in preprocessed_records:
        logger.info(f"[{idx+1}/{len(preprocessed_records)}] Classifying ID: {payload['id']}...")
        result = classifier.classify(payload)
        categories.append(result.category)
        confidences.append(result.confidence)
        reasonings.append(result.reasoning)
        
    # Create output dataframe
    output_df = df_to_process.copy()
    output_df["classification_category"] = categories
    output_df["confidence"] = confidences
    output_df["reasoning"] = reasonings
    
    logger.info(f"Writing results to {args.output}")
    try:
        output_df.to_csv(args.output, index=False)
        logger.info("Processing complete!")
    except Exception as e:
        logger.error(f"Failed to write output CSV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
