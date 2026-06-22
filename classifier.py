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
    def safe_str(field_name: str, fallback_field: Optional[str] = None) -> str:
        val = row.get(field_name)
        if pd.isna(val) or val is None:
            if fallback_field:
                return safe_str(fallback_field)
            return ""
        return str(val).strip()

    listing_id = safe_str("id")
    summary = safe_str("summary", "name")
    description = safe_str("detailedDescription", "description")
    
    raw_kf = row.get("keyFeatures")
    kf_list = parse_key_features(raw_kf) if pd.notna(raw_kf) else []
    key_features = ", ".join(kf_list) if kf_list else ""
    
    use_class = safe_str("useClass")
    address = safe_str("displayAddress", "address")
    subtype = safe_str("propertySubType", "subType")
    
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
        self.mock_mode = False
        
        # Check if we should activate Mock Mode
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment. The script will run in MOCK MODE for local verification.")
            self.mock_mode = True
            
    def _get_client(self):
        """Lazy load OpenAI client so API key isn't required for dry-runs."""
        if self.mock_mode:
            return None
        if self.client is None:
            if not self.api_key:
                raise ValueError("OpenAI API Key is missing. Please set OPENAI_API_KEY in your environment or .env file.")
            self.client = OpenAI(api_key=self.api_key)
        return self.client

    def _get_mock_classification(self, listing_data: dict) -> PropertyClassification:
        """
        Simulates the gpt-4o-mini classification results for local testing and validation when API key is absent.
        Uses a simple, generic keyword-based heuristic rather than hardcoded IDs or dataset-specific rules.
        """
        text_payload = (
            f"{listing_data['summary']} {listing_data['description']} "
            f"{listing_data['key_features']} {listing_data['property_subtype']}"
        ).lower()
        
        if any(kw in text_payload for kw in ["nursery", "daycare", "childcare", "day nursery", "play area"]):
            confidence = "High" if "d1" in text_payload or "f1" in text_payload or "nursery consent" in text_payload else "Medium"
            return PropertyClassification(
                category="Nursery",
                confidence=confidence,
                reasoning="Mock: Property listing mentions childcare, nursery, or daycare facilities with potential target features."
            )
        elif any(kw in text_payload for kw in ["sen school", "special educational needs", "complex needs"]):
            confidence = "High" if "let to" in text_payload or "grammar school" in text_payload else "Medium"
            return PropertyClassification(
                category="SEN School",
                confidence=confidence,
                reasoning="Mock: Property mentions special educational needs or grammar school operations/facilities."
            )
        elif any(kw in text_payload for kw in ["co-op", "convenience store", "supermarket", "costcutter", "grocery"]):
            confidence = "High" if "former convenience" in text_payload or "retail unit" in text_payload else "Medium"
            return PropertyClassification(
                category="Food Store",
                confidence=confidence,
                reasoning="Mock: Listing references food retail, convenience store, or supermarket operations."
            )
            
        return PropertyClassification(
            category="None",
            confidence="High",
            reasoning="Mock: Listing describes a general commercial asset (industrial/office/retail/land) with no keywords aligning with target sectors."
        )

    def classify(self, listing_data: dict, max_retries: int = 3, initial_delay: float = 2.0) -> PropertyClassification:
        """
        Classifies a single property listing using OpenAI's Structured Outputs.
        Falls back to Mock Mode if OPENAI_API_KEY is missing.
        """
        if self.mock_mode:
            # Simulate a small networks delay to make the CLI feel realistic
            time.sleep(0.1)
            return self._get_mock_classification(listing_data)
            
        from prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

        client = self._get_client()
        user_content = USER_PROMPT_TEMPLATE.format(
            id=listing_data["id"],
            address=listing_data["address"],
            property_subtype=listing_data["property_subtype"],
            use_class=listing_data["use_class"],
            summary=listing_data["summary"],
            key_features=listing_data["key_features"],
            points_of_interest=listing_data["points_of_interest"],
            description=listing_data["description"]
        )

        delay = initial_delay
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                response = client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    response_format=PropertyClassification
                )
                
                parsed = response.choices[0].message.parsed
                if parsed:
                    return parsed
                else:
                    raise ValueError("OpenAI API returned an empty structured response.")
                    
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"API call failed for ID {listing_data['id']} (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    logger.info(f"Waiting {delay:.1f} seconds before retrying...")
                    time.sleep(delay)
                    delay *= 2.0

        logger.error(
            f"All {max_retries} attempts failed for ID {listing_data['id']}. Falling back to default None classification."
        )
        return PropertyClassification(
            category="None",
            confidence="Low",
            reasoning="Classification could not be completed due to an API error."
        )
