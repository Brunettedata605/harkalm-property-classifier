import ast
import logging
import os
import time
from typing import Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI
from utils import parse_key_features

logger = logging.getLogger(__name__)

# Pydantic schema for OpenAI Structured Outputs
class PropertyClassification(BaseModel):
    category: Literal["Nursery", "SEN School", "Food Store", "None"] = Field(
        description="The classified category of the property listing."
    )
    confidence: Literal["High", "Medium", "Low"] = Field(
        description="Confidence level in the classification. Use Low/Medium if the listing is speculative, hedges on use, or has weak evidence."
    )
    reasoning: str = Field(
        description="A short, professional explanation of the decision, citing specific details from the listing."
    )

def parse_points_of_interest(val: str) -> str:
    """Parses pointsOfInterest stringified lists of dicts into a clean string representation."""
    if pd.isna(val) or not val or not isinstance(val, str):
        return ""
    val_str = val.strip()
    if not val_str:
        return ""
    try:
        parsed = ast.literal_eval(val_str)
        if isinstance(parsed, list):
            poi_items = []
            for item in parsed:
                if isinstance(item, dict):
                    title = item.get("title") or item.get("name")
                    poi_type = item.get("type") or item.get("__typename") or ""
                    dist = item.get("distanceMiles") or item.get("distance")
                    unit = item.get("unit") or "miles"
                    
                    if title:
                        dist_info = f" ({dist} {unit})" if dist is not None else ""
                        type_info = f" [{poi_type}]" if poi_type else ""
                        poi_items.append(f"{title}{type_info}{dist_info}")
            return ", ".join(poi_items)
    except Exception as e:
        logger.debug(f"Failed to parse pointsOfInterest with ast.literal_eval: {e}")
    return str(val)

def preprocess_row(row: pd.Series) -> dict:
    """
    Extracts and sanitizes relevant fields from a row, creating a clean dictionary
    representing the property listing metadata.
    """
    # Safe string extraction with fallback to empty string
    def safe_str(field_name: str, fallback_field: Optional[str] = None) -> str:
        val = row.get(field_name)
        if pd.isna(val) or val is None:
            if fallback_field:
                return safe_str(fallback_field)
            return ""
        return str(val).strip()

    # Extract ID
    listing_id = safe_str("id")
    
    # Extract summary
    summary = safe_str("summary", "name")
    
    # Extract description
    description = safe_str("detailedDescription", "description")
    
    # Extract key features
    raw_kf = row.get("keyFeatures")
    kf_list = parse_key_features(raw_kf) if pd.notna(raw_kf) else []
    key_features = ", ".join(kf_list) if kf_list else ""
    
    # Extract use class
    use_class = safe_str("useClass")
    
    # Extract address
    address = safe_str("displayAddress", "address")
    
    # Extract property subtype
    subtype = safe_str("propertySubType", "subType")
    
    # Extract points of interest
    raw_poi = row.get("pointsOfInterest")
    poi_str = parse_points_of_interest(raw_poi) if pd.notna(raw_poi) else ""
    
    return {
        "id": listing_id,
        "summary": summary,
        "description": description,
        "key_features": key_features,
        "use_class": use_class,
        "address": address,
        "property_subtype": subtype,
        "points_of_interest": poi_str
    }

class PropertyClassifier:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initializes the OpenAI client and configurations."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", model)
        self.client = None
        
    def _get_client(self):
        """Lazy load OpenAI client so API key isn't required for dry-runs."""
        if self.client is None:
            if not self.api_key:
                raise ValueError("OpenAI API Key is missing. Please set OPENAI_API_KEY in your environment or .env file.")
            self.client = OpenAI(api_key=self.api_key)
        return self.client

    def classify(self, listing_data: dict, max_retries: int = 3, initial_delay: float = 2.0) -> PropertyClassification:
        """
        Classifies a single property listing using OpenAI's Structured Outputs.
        Includes exponential backoff retry logic for resilience.
        """
        # We will implement this in Milestone 3
        # Returning a placeholder classification for Milestone 2 dry-run/verifications
        return PropertyClassification(
            category="None",
            confidence="High",
            reasoning="LLM API client call stub. (Dry Run / Implementation placeholder)"
        )
