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
from datetime import datetime
from utils.logger import color_logger
from openai import OpenAI

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '../../../../.env')
load_dotenv(env_path)

# Initialize variables
pc = None
index = None
index_name = "ai-world-predictions-pinecone-dave"

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
            model="text-embedding-ada-002",
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
        'model': 'text-embedding-ada-002',
        'timestamp': datetime.now().isoformat()
    }))
except Exception as e:
    color_logger.error(json.dumps({
        'component': 'model_setup',
        'status': 'error',
        'model': 'text-embedding-ada-002',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }))

# def upsert_data_to_pinecone(data):
#     """Upsert data to Pinecone with proper metadata"""
#     try:
#         if not data:
#             raise ValueError("No data provided for upserting")
#         vectors = []
#         for i, item in enumerate(data):
#             embedding = text_to_vector(item)
#             vectors.append(
#                 (f'vec_{i}', embedding, {"text": item, "source": "firecrawl"})
#             )

#         index.upsert(vectors)
#         color_logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")

#     except Exception as e:
#         raise RuntimeError(f"Failed to upsert data to Pinecone: {str(e)}")

def upsert_data_to_pinecone(data):
    """Upsert data to Pinecone with proper metadata"""
    try:
        if not data:
            raise ValueError("No data provided for upserting")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vectors = []
        for i, item in enumerate(data):
            embedding = text_to_vector(item)
            # Create unique ID with timestamp
            unique_id = f'vec_{timestamp}_{i}'
            vectors.append(
                (unique_id, embedding, {"text": item, "source": "firecrawl", "timestamp": timestamp})
            )
        index.upsert(vectors)
        color_logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
    except Exception as e:
        raise RuntimeError(f"Failed to upsert data to Pinecone: {str(e)}")

if __name__ == "__main__":
    # Test query after initialization
    test_result = query_pinecone("test query")
    color_logger.info(f"Test query results: {json.dumps(test_result, indent=2)}")

def process_and_index_data(data_type="default"):
    """
    Read JSON data from file and index it in Pinecone
    Args:
        json_filename (str): Name of the JSON file to process
        data_type (str): Type of data to process ("sports", "default", etc.)
    """
    try:
        # Read JSON file
        current_dir = os.path.dirname(__file__)
        outputs_path = os.path.join(current_dir, '..', '..', 'services', 'webScraping', 'outputs', 'sports_data.json')

        with open(outputs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract relevant text fields for indexing
        texts_to_index = []

        if data_type.lower() == "sports":
            for article in data:
                parts = []

                # Add title and summary
                if article.get('title'):
                    parts.append(f"Title: {article['title']}")
                if article.get('summary'):
                    parts.append(f"Summary: {article['summary']}")

                # Add player information
                if article.get('players'):
                    for player in article['players']:
                        player_info = []
                        if player.get('name'):
                            player_info.append(f"Player: {player['name']}")
                        if player.get('team'):
                            player_info.append(f"Team: {player['team']}")
                        if player.get('position'):
                            player_info.append(f"Position: {player['position']}")
                        if player.get('stats'):
                            player_info.append(f"Stats: {', '.join(player['stats'])}")
                        if player_info:
                            parts.append(" | ".join(player_info))

                # Add team information
                if article.get('teams'):
                    for team in article['teams']:
                        team_info = []
                        if team.get('name'):
                            team_info.append(f"Team: {team['name']}")
                        if team.get('stats'):
                            team_info.append(f"Team Stats: {', '.join(team['stats'])}")
                        if team_info:
                            parts.append(" | ".join(team_info))

                # Add quote if it exists
                if article.get('quote'):
                    parts.append(f"Quote: {article['quote']}")

                # Combine all parts with newlines
                article_text = "\n".join(parts)

                if article_text.strip():
                    texts_to_index.append(article_text)

        else:  # default handling for other data types
            for article in data:
                article_text = f"{article.get('title', '')} {article.get('summary', '')}"
                if article_text.strip():
                    texts_to_index.append(article_text)

        # Index in Pinecone
        if texts_to_index:
            print(f"Indexing {len(texts_to_index)} articles in Pinecone...")
            upsert_data_to_pinecone(texts_to_index)
            print("Successfully indexed articles in Pinecone")

            # Optionally delete the JSON file after successful indexing
            os.remove(outputs_path)
            print(f"Deleted sports_data.json after successful indexing")

    except Exception as e:
        color_logger.warning(json.dumps({
            'component': 'pinecone_query',
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }))

        # Return mock data as fallback
        return {
            "matches": [
                {
                    "metadata": {
                        "text": "Warriors have won 7 of their last 10 games against the Lakers"
                    }
                },
                {
                    "metadata": {
                        "text": "Historical head-to-head record favors Warriors when Curry is healthy"
                    }
                }
            ]
        }

@color_logger.log_service_call('pinecone')
def query_pinecone(query_text):
    """Query Pinecone with fallback to mock data"""
    try:
        if not query_text:
            raise ValueError("Query text cannot be empty")

        color_logger.info(json.dumps({
            'component': 'pinecone_query',
            'status': 'start',
            'query_length': len(query_text),
            'timestamp': datetime.now().isoformat()
        }))

        if pc and index:
            # For now, return mock data even if we have a connection
            color_logger.info(json.dumps({
                'component': 'pinecone_query',
                'status': 'success',
                'message': 'Using mock data for development',
                'timestamp': datetime.now().isoformat()
            }))

        # Return mock data
        return {
            "matches": [
                {
                    "metadata": {
                        "text": "Warriors have won 7 of their last 10 games against the Lakers"
                    }
                },
                {
                    "metadata": {
                        "text": "Historical head-to-head record favors Warriors when Curry is healthy"
                    }
                }
            ]
        }

    except Exception as e:
        color_logger.warning(json.dumps({
            'component': 'pinecone_query',
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }))

        # Return mock data as fallback
        return {
            "matches": [
                {
                    "metadata": {
                        "text": "Warriors have won 7 of their last 10 games against the Lakers"
                    }
                },
                {
                    "metadata": {
                        "text": "Historical head-to-head record favors Warriors when Curry is healthy"
                    }
                }
            ]
        }

def process_and_index_data(data_type="default"):
    """
    Read JSON data from file and index it in Pinecone
    Args:
        json_filename (str): Name of the JSON file to process
        data_type (str): Type of data to process ("sports", "default", etc.)
    """
    try:
        # Read JSON file
        current_dir = os.path.dirname(__file__)
        outputs_path = os.path.join(current_dir, '..', '..', 'services', 'webScraping', 'outputs', 'sports_data.json')

        with open(outputs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract relevant text fields for indexing
        texts_to_index = []

        if data_type.lower() == "sports":
            for article in data:
                parts = []

                # Add title and summary
                if article.get('title'):
                    parts.append(f"Title: {article['title']}")
                if article.get('summary'):
                    parts.append(f"Summary: {article['summary']}")

                # Add player information
                if article.get('players'):
                    for player in article['players']:
                        player_info = []
                        if player.get('name'):
                            player_info.append(f"Player: {player['name']}")
                        if player.get('team'):
                            player_info.append(f"Team: {player['team']}")
                        if player.get('position'):
                            player_info.append(f"Position: {player['position']}")
                        if player.get('stats'):
                            player_info.append(f"Stats: {', '.join(player['stats'])}")
                        if player_info:
                            parts.append(" | ".join(player_info))

                # Add team information
                if article.get('teams'):
                    for team in article['teams']:
                        team_info = []
                        if team.get('name'):
                            team_info.append(f"Team: {team['name']}")
                        if team.get('stats'):
                            team_info.append(f"Team Stats: {', '.join(team['stats'])}")
                        if team_info:
                            parts.append(" | ".join(team_info))

                # Add quote if it exists
                if article.get('quote'):
                    parts.append(f"Quote: {article['quote']}")

                # Combine all parts with newlines
                article_text = "\n".join(parts)

                if article_text.strip():
                    texts_to_index.append(article_text)

        else:  # default handling for other data types
            for article in data:
                article_text = f"{article.get('title', '')} {article.get('summary', '')}"
                if article_text.strip():
                    texts_to_index.append(article_text)

        # Index in Pinecone
        if texts_to_index:
            print(f"Indexing {len(texts_to_index)} articles in Pinecone...")
            upsert_data_to_pinecone(texts_to_index)
            print("Successfully indexed articles in Pinecone")

            # Optionally delete the JSON file after successful indexing
            os.remove(outputs_path)
            print(f"Deleted sports_data.json after successful indexing")

    except Exception as e:
        print(f"Error processing and indexing data: {e}")
