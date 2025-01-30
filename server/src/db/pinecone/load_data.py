import os
import sys
import json
from datetime import datetime
import openai
from typing import List, Dict, Any

# Add the server/src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from db.pinecone.setup_pinecone import initialize_pinecone, index
from utils.logger import color_logger

def get_embedding(text: str) -> List[float]:
    """Get embedding using OpenAI's text-embedding-ada-002 model"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        color_logger.error(f"Error getting embedding: {str(e)}")
        raise

def safe_join(items: Any) -> str:
    """Safely join items into a string, handling various data types"""
    if isinstance(items, list):
        return ', '.join(str(item) for item in items)
    elif items is None:
        return ''
    return str(items)

def load_json_data(file_path: str, source_type: str):
    """Load data from a JSON file into Pinecone"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        color_logger.info(f"Loading {len(data)} {source_type} articles into Pinecone")

        # Convert articles to vectors and metadata
        vectors = []
        for i, article in enumerate(data):
            # Create a comprehensive text representation based on source type
            if source_type == 'sports':
                text = f"""
                Title: {article.get('title', '')}
                Content: {article.get('content', '')}
                Teams: {safe_join(article.get('teams', []))}
                League: {article.get('league', '')}
                Date: {article.get('date', '')}
                """
                metadata = {
                    'source': f'{source_type}_crawler',
                    'title': article.get('title', ''),
                    'date': article.get('date', ''),
                    'teams': safe_join(article.get('teams', [])),
                    'league': article.get('league', ''),
                    'text': text[:1000]
                }
            elif source_type == 'finance':
                text = f"""
                Title: {article.get('title', '')}
                Content: {article.get('content', '')}
                Companies: {safe_join(article.get('companies', []))}
                Market: {article.get('market', '')}
                Date: {article.get('date', '')}
                """
                metadata = {
                    'source': f'{source_type}_crawler',
                    'title': article.get('title', ''),
                    'date': article.get('date', ''),
                    'companies': safe_join(article.get('companies', [])),
                    'market': article.get('market', ''),
                    'text': text[:1000]
                }
            else:  # Default format for other types
                text = f"""
                Title: {article.get('title', '')}
                Content: {article.get('content', '')}
                Date: {article.get('date', '')}
                """
                metadata = {
                    'source': f'{source_type}_crawler',
                    'title': article.get('title', ''),
                    'date': article.get('date', ''),
                    'text': text[:1000]
                }

            # Get embedding using OpenAI
            vector = get_embedding(text)
            vectors.append((f'{source_type}_{i}', vector, metadata))

            if (i + 1) % 10 == 0:
                color_logger.info(f"Processed {i + 1} {source_type} articles")

        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            color_logger.info(f"Uploaded batch {i//batch_size + 1} of {source_type} data")

        color_logger.info(f"✅ Successfully loaded {source_type} data into Pinecone")
        return True

    except Exception as e:
        color_logger.error(f"Error loading {source_type} data: {str(e)}")
        return False

def load_all_data():
    """Load all JSON files from the outputs directory"""
    outputs_dir = os.path.join(os.path.dirname(__file__), '../../services/webScraping/outputs')

    # Map of file names to their source types
    file_types = {
        'sports_data.json': 'sports',
        'finance_data.json': 'finance',
        'entertainment_data.json': 'entertainment',
        'us_news_data.json': 'news'
    }

    success_count = 0
    total_files = len(file_types)

    for filename, source_type in file_types.items():
        file_path = os.path.join(outputs_dir, filename)
        color_logger.info(f"\n🔄 Processing {filename}...")

        if os.path.exists(file_path):
            if load_json_data(file_path, source_type):
                success_count += 1
        else:
            color_logger.warning(f"File not found: {filename}")

    color_logger.info(f"\n📊 Summary: Successfully loaded {success_count}/{total_files} data files")
    return success_count == total_files

if __name__ == "__main__":
    # Initialize Pinecone
    if initialize_pinecone():
        # Load all data
        if load_all_data():
            color_logger.info("🎉 All data loading complete!")
        else:
            color_logger.warning("⚠️ Some data files failed to load")
    else:
        color_logger.error("❌ Failed to initialize Pinecone")
