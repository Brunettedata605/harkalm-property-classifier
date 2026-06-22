import ast
import logging

def setup_logging():
    """Sets up a clean, professional logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s - %(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def parse_key_features(val: str) -> list[str]:
    """
    Safely parses keyFeatures, which is represented as a stringified list in the CSV.
    Uses ast.literal_eval for safe evaluation, falling back to manual split if needed.
    """
    if not val or not isinstance(val, str):
        return []
    
    val = val.strip()
    if not val:
        return []
        
    try:
        # ast.literal_eval is safe to evaluate strings representing Python literals/structures
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if item]
        return []
    except (ValueError, SyntaxError) as e:
        # Log debug info and try simple split fallback
        logging.debug(f"ast.literal_eval failed on keyFeatures raw value: '{val}'. Error: {e}. Falling back to split.")
        cleaned = val.strip("[]")
        # Split by comma and clean up quotes
        items = [item.strip().strip("'\"") for item in cleaned.split(",")]
        return [item for item in items if item]
