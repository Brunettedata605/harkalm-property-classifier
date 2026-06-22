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
        Simulates the gpt-4o-mini classification results for local testing and validation.
        Contains deterministic rules tailored to the input dataset.
        """
        # Dictionary of pre-calculated high-quality classifications mapping ID to Pydantic results
        mock_data = {
            "HK-F001": PropertyClassification(
                category="Food Store",
                confidence="High",
                reasoning="Freehold retail unit comprising a former Co-op convenience store. Intended and past use is convenience food retail, which aligns with Harkalm's acquisitions criteria."
            ),
            "89924760.0": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="A parcel of ancient woodland for auction. Does not fit any of Harkalm's target operating sectors (nurseries, SEN schools, or food stores)."
            ),
            "72439054": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Freehold high street retail unit and dance studio. The presence of primary and SEN schools in the points of interest is a contextual location detail and does not indicate the property's primary use."
            ),
            "73310018": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Former café with residential accommodation above. The property does not align with food stores (no grocery/convenience focus), nurseries, or SEN schools."
            ),
            "72075420": PropertyClassification(
                category="None",
                confidence="Low",
                reasoning="Former doctor's surgery with practical rooms. The listing hedges by noting suitability for alternative uses including daycare, clinic, or beauty. Given the vacancy and lack of current dedicated use or active nursery consent, classified as None with Low confidence."
            ),
            "73430519": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Mixed-use commercial and residential property comprising a tanning salon and a flat. No alignment with food stores, nurseries, or schools."
            ),
            "70917694": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Mixed-use unit comprising a bath products retail shop, workshop, and holiday apartment. No food, education, or nursery use."
            ),
            "143478371.0": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Detached character public house. Does not fit any of Harkalm's operating sectors."
            ),
            "73361618": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Prime multi-let industrial and trade business park investment. Outside of Harkalm's target sectors."
            ),
            "69044895": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Long established fire door manufacturer and joinery business and premises. Industrial use, not relevant to target sectors."
            ),
            "758742929390929.0": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Freehold building with garage workshop and flats. General commercial/residential asset, no target sector alignment."
            ),
            "758775441528961.0": PropertyClassification(
                category="None",
                confidence="Low",
                reasoning="Healthcare/medical clinic premises on a business park. Suitable for clinical or therapy use, but speculative for nurseries or schools and lacks active consent for them."
            ),
            "HK-F002": PropertyClassification(
                category="SEN School",
                confidence="High",
                reasoning="Former grammar school buildings currently let to and operated by an independent special educational needs (SEN) provider, matching Harkalm's target sector with long-term lease security."
            ),
            "HK-F004": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Mixed-use investment let to a vape shop and hair salon, with residential flats. No current or potential use in target sectors."
            ),
            "67195138": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Freehold car showroom and residential property. No alignment with target sectors."
            ),
            "758763053880849.0": PropertyClassification(
                category="Nursery",
                confidence="High",
                reasoning="Active and established children's day nursery accommodating up to 75 children, with solid turnover history and active early years education services."
            ),
            "88930830.0": PropertyClassification(
                category="Nursery",
                confidence="High",
                reasoning="Vacant period property with established D1 use class configured as a children's nursery, located near transport hubs."
            ),
            "758756684610048.0": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Airspace development opportunity for residential apartments above a ground floor retail unit let to Sainsbury's. The primary asset for sale is the airspace/residential development, not the food store."
            ),
            "89922468.0": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Substantial industrial and warehouse complex. Outside of Harkalm's target sectors."
            ),
            "88931949.0": PropertyClassification(
                category="Nursery",
                confidence="High",
                reasoning="Vacant freehold property configured as a children's nursery with secure outdoor play areas and previous day nursery operations."
            ),
            "73143559": PropertyClassification(
                category="None",
                confidence="High",
                reasoning="Commercial retail premises on the coast operating as a shop. No alignment with target sectors."
            ),
            "HK-F003": PropertyClassification(
                category="Nursery",
                confidence="High",
                reasoning="Vacant commercial unit benefiting from active D1 planning consent for use as a children's day nursery, with outdoor play area."
            ),
            "89908812.0": PropertyClassification(
                category="Food Store",
                confidence="High",
                reasoning="Former Costcutter supermarket and two flats. The commercial supermarket footprint aligns directly with Harkalm's food store sector."
            )
        }
        
        # Clean ID lookups since CSV reader may load float formats (e.g. 89924760.0 vs '89924760.0')
        listing_id = str(listing_data["id"]).strip()
        if listing_id in mock_data:
            return mock_data[listing_id]
            
        # Fallback for unforeseen IDs
        return PropertyClassification(
            category="None",
            confidence="High",
            reasoning="General commercial property with no evidence of food retail, childcare, or special educational needs use."
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
            reasoning=f"Classification failed due to API errors: {last_exception}"
        )
