import os
import sys
from datetime import datetime
from typing import Dict

# Add the server/src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pinecone import Pinecone
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import json
from utils.logger import color_logger
from openai import OpenAI

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '../../../../.env')
load_dotenv(env_path)

# Initialize variables
pc = None
index = None
index_name = "ai-world-predictions-pinecone"

@color_logger.log_service_call('pinecone')
def query_pinecone(query_text: str, filter_params: Dict = None):
    """Query Pinecone with proper error handling and logging"""
    try:
        if not query_text:
            raise ValueError("Query text cannot be empty")

        color_logger.info(json.dumps({
            'component': 'pinecone_query',
            'status': 'start',
            'query': query_text[:100],  # Log first 100 chars of query
            'timestamp': datetime.now().isoformat()
        }))

        if not pc or not index:
            # Try to initialize Pinecone if not already initialized
            if not initialize_pinecone():
                raise RuntimeError("Failed to initialize Pinecone connection")

        # Convert query to vector
        query_vector = text_to_vector(query_text)

        # Perform the query with filters if provided
        query_params = {
            'vector': query_vector,
            'top_k': 10,  # Increased from 5 to get more results
            'include_metadata': True
        }

        if filter_params:
            query_params['filter'] = filter_params
        else:
            # Add default filter for NFL/Super Bowl related content
            query_params['filter'] = {
                'league': {'$in': ['NFL']},
                'event_type': {'$in': ['Super Bowl', 'NFL Game', 'NFL Prediction']}
            }

        results = index.query(**query_params)

        # Log success and return results
        match_count = len(results.get('matches', []))
        color_logger.info(json.dumps({
            'component': 'pinecone_query',
            'status': 'success',
            'matches_found': match_count,
            'timestamp': datetime.now().isoformat()
        }))

        if match_count == 0:
            color_logger.warning("No matches found with filters - trying broader search")
            # Try again without filters
            results = index.query(
                vector=query_vector,
                top_k=10,
                include_metadata=True
            )
            match_count = len(results.get('matches', []))

            if match_count == 0:
                raise ValueError("No matches found in Pinecone index even with broader search")

        return results

    except Exception as e:
        color_logger.error(f"Pinecone query failed: {str(e)}")
        raise  # Re-raise the exception instead of returning mock data

def get_mock_data():
    """Return mock data with clear indication it's not from Pinecone"""
    return {
        "matches": [
            {
                "metadata": {
                    "text": "[MOCK DATA] Warriors have won 7 of their last 10 games against the Lakers",
                    "source": "mock_data",
                    "timestamp": datetime.now().isoformat()
                }
            },
            {
                "metadata": {
                    "text": "[MOCK DATA] Historical head-to-head record favors Warriors when Curry is healthy",
                    "source": "mock_data",
                    "timestamp": datetime.now().isoformat()
                }
            }
        ]
    }

def text_to_vector(text):
    """Convert text to vector using OpenAI embeddings"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        # Get the embedding vector
        embedding = response.data[0].embedding
        return embedding  # This will be 1536-dimensional
    except Exception as e:
        color_logger.error(json.dumps({
            'component': 'text_to_vector',
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }))
        # Return zero vector of correct dimension (1536)
        return [0] * 1536

# Initialize Pinecone
def initialize_pinecone():
    """Initialize Pinecone with proper error handling"""
    global pc, index

    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)

        # Test the connection
        index.describe_index_stats()

        color_logger.info("Pinecone initialized successfully")
        return True

    except Exception as e:
        color_logger.error(f"Failed to initialize Pinecone: {str(e)}")
        return False

# Call initialization on module load
is_pinecone_ready = initialize_pinecone()

# Load OpenAI client
try:
    openai_client = OpenAI()
    color_logger.info(json.dumps({
        'component': 'model_setup',
        'status': 'success',
        'model': 'text-embedding-3-small',
        'timestamp': datetime.now().isoformat()
    }))
except Exception as e:
    color_logger.error(json.dumps({
        'component': 'model_setup',
        'status': 'error',
        'model': 'text-embedding-3-small',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }))

def upsert_data_to_pinecone(data):
    """Upsert data to Pinecone with proper metadata"""
    try:
        if not data:
            raise ValueError("No data provided for upserting")

        vectors = []
        for i, item in enumerate(data):
            # Ensure item is a dictionary with required fields
            if isinstance(item, dict):
                text = item.get('text', '')
                metadata = {
                    'source': item.get('source', 'firecrawl'),
                    'league': item.get('league', 'NFL'),
                    'event_type': item.get('event_type', 'NFL Game'),
                    'timestamp': datetime.now().isoformat()
                }
                metadata.update({k: v for k, v in item.items() if k not in ['text', 'source']})
            else:
                text = str(item)
                metadata = {
                    'source': 'firecrawl',
                    'league': 'NFL',
                    'event_type': 'NFL Game',
                    'timestamp': datetime.now().isoformat()
                }

            vector = text_to_vector(text)
            vectors.append((f'vec_{i}', vector, metadata))

        index.upsert(vectors)
        color_logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")

    except Exception as e:
        raise RuntimeError(f"Failed to upsert data to Pinecone: {str(e)}")

if __name__ == "__main__":
    # Test query after initialization
    test_result = query_pinecone("test query")
    color_logger.info(f"Test query results: {json.dumps(test_result, indent=2)}")
