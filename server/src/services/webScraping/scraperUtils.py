from typing import Any, Dict, List, Type, get_type_hints
import json
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field, create_model
from .models import DataPoints
from tenacity import retry, wait_random_exponential, stop_after_attempt

def create_filtered_model(data: List[Dict[str, Any]], base_model: Type[BaseModel], links_scraped: List[str]) -> Type[BaseModel]:
    # Filter fields where value is None
    filtered_fields = {item['name']: item['value'] for item in data if item['value'] is None or isinstance(item['value'], list)}

    # Get fields with their annotations and descriptions
    fields_with_descriptions = {
        field: (base_model.__annotations__[field], Field(..., description=base_model.__fields__[field].description))
        for field in filtered_fields
    }

    # Constructing the desired JSON output
    data_to_collect = [
        {"name": field_name, "description": field_info.description}
        for field_name, (field_type, field_info) in fields_with_descriptions.items()
    ]

    print(f"Fields with descriptions: {data_to_collect}")
    # Create and return new Pydantic model
    FilteredModel = create_model('FilteredModel', **fields_with_descriptions)

    ExtendedDataPoints = create_model(
        'DataPoints',
        relevant_urls_might_contain_further_info=(List[str], Field([], description=f"Relevant urls that we should scrape further that might contain information related to data points that we want to find; [Articles] {data_to_collect} [/END DATA POINTS] Prioritise urls on official their own domain first, even file url of image or pdf - those links can often contain useful information, we should always prioritise those urls instead of external ones; return None if cant find any; links cannot be any of the following: {links_scraped}")),
        __base__=FilteredModel
    )

    return ExtendedDataPoints

async def extract_data_from_content(content: str, data_points: List[Dict], links_scraped: List[str], url: str) -> Dict:
    FilteredModel = create_filtered_model(data_points, DataPoints, links_scraped)
    client = instructor.from_openai(OpenAI())

    result = await client.chat.completions.create(
        model="gpt-4o",
        response_model=FilteredModel,
        messages=[{"role": "user", "content": content}],
    )

    filtered_data = filter_empty_fields(result)

    data_to_update = [
        {"name": key, "value": value["value"], "reference": url, "type": value["type"]}
        for key, value in filtered_data.items() if key != 'relevant_urls_might_contain_further_info'
    ]

    from .scraper import update_data  # Updated to relative import
    update_data(data_points, data_to_update)

    return result.json()

def filter_empty_fields(model_instance: BaseModel) -> dict:
    def _filter(data: Any, field_type: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: _filter(v, field_type.get(k, type(v)) if isinstance(field_type, dict) else type(v))
                for k, v in data.items()
                if v not in [None, "", [], {}, "null", "None"]
            }
        elif isinstance(data, list):
            return [
                _filter(item, field_type.__args__[0] if hasattr(field_type, '__args__') else type(item))
                for item in data
                if item not in [None, "", [], {}, "null", "None"]
            ]
        else:
            return data

    data_dict = model_instance.dict(exclude_none=True)
    field_types = get_type_hints(model_instance.__class__)

    def get_inner_type(field_type):
        if hasattr(field_type, '__origin__') and field_type.__origin__ == list:
            return list
        return field_type

    filtered_dict = {
        k: {
            "value": _filter(v, get_inner_type(field_types.get(k, type(v)))),
            "type": str(get_inner_type(field_types.get(k, type(v))).__name__)
        }
        for k, v in data_dict.items()
        if v not in [None, "", [], {}, "null", "None"]
    }

    return filtered_dict
