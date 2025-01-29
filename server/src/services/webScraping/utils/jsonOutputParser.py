import os
import json

def save_json_pretty(data, filename):
    """
    Append data to an existing JSON file or create a new one if it doesn't exist.
    
    Args:
        data: The new data to append (can be Pydantic model or dict)
        filename (str): The name of the file
    """
    try:
        # Set path to outputs directory
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', filename)
        
        # Convert Pydantic model to dict if necessary
        if hasattr(data, 'model_dump'):  # For Pydantic v2
            data = data.model_dump()
        elif hasattr(data, 'dict'):      # For Pydantic v1
            data = data.dict()
            
        # Initialize existing_data as an empty list
        existing_data = []
        
        # Try to read existing file
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except FileNotFoundError:
            print(f"Creating new file: {filepath}")
        except json.JSONDecodeError:
            print(f"Error reading existing file. Creating new file: {filepath}")
            
        # Append new data
        if isinstance(data, list):
            existing_data.extend(data)
        else:
            existing_data.append(data)
            
        # Write back to file
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=4, sort_keys=True, ensure_ascii=False)
        print(f"Data successfully appended to {filepath}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
