from pinecone import Pinecone
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import json
from datetime import datetime
from utils.logger import color_logger

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '../../../../.env')
load_dotenv(env_path)

# Initialize variables
pc = None
index = None
index_name = "ai-world-predictions-2"

@color_logger.log_service_call('pinecone')
def query_pinecone(query_text: str):
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
            query_vector = text_to_vector(query_text)
            results = index.query(query_vector, top_k=5, include_metadata=True)

            color_logger.info(json.dumps({
                'component': 'pinecone_query',
                'status': 'success',
                'matches_found': len(results.get('matches', [])),
                'timestamp': datetime.now().isoformat()
            }))

            return results
        else:
            raise RuntimeError("Pinecone not initialized")

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

def text_to_vector(text):
    """Convert text to vector with error handling"""
    try:
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        vector = torch.mean(outputs.last_hidden_state, dim=1).squeeze().tolist()
        return vector
    except Exception as e:
        color_logger.warning(json.dumps({
            'component': 'text_to_vector',
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }))
        return [0] * 768  # Return zero vector as fallback

# Initialize Pinecone
try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key:
        pc = Pinecone(api_key=pinecone_api_key)
        try:
            index = pc.Index(index_name)
            color_logger.info(json.dumps({
                'component': 'pinecone_setup',
                'status': 'success',
                'message': 'Connected to existing index',
                'timestamp': datetime.now().isoformat()
            }))
        except Exception as e:
            color_logger.warning(json.dumps({
                'component': 'pinecone_setup',
                'status': 'creating_index',
                'message': f'Index {index_name} not found, using mock data',
                'timestamp': datetime.now().isoformat()
            }))
    else:
        color_logger.warning(json.dumps({
            'component': 'pinecone_setup',
            'status': 'no_api_key',
            'message': 'PINECONE_API_KEY not found, will use mock data',
            'timestamp': datetime.now().isoformat()
        }))
except Exception as e:
    color_logger.warning(json.dumps({
        'component': 'pinecone_setup',
        'status': 'error',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }))

# Load BERT model
try:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    color_logger.info(json.dumps({
        'component': 'bert_setup',
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    }))
except Exception as e:
    color_logger.warning(json.dumps({
        'component': 'bert_setup',
        'status': 'error',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }))

def upsert_data_to_pinecone(data):
    try:
        if not data:
            raise ValueError("No data provided for upserting")
        vectors = []
        for i, item in enumerate(data):
            embedding = text_to_vector(item)
            vectors.append(
                (f'vec_{i}', embedding, {"text": item, "source": "firecrawl"})
            )

        index.upsert(vectors)
    except Exception as e:
        raise RuntimeError(f"Failed to upsert data to Pinecone: {str(e)}")

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
        
        with open(outputs_path, 'r') as f:
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
            os.remove(os.path.join('outputs', "sports_data.json"))
            print(f"Deleted sports_data.json after successful indexing")
            
    except Exception as e:
        print(f"Error processing and indexing data: {e}")