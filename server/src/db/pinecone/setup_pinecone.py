from pinecone import Pinecone
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '../../../../.env')
load_dotenv(env_path)

# Initialize variables
pc = None
index = None

try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key:
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "ai-world-predictions-pinecone"
        index = pc.Index(index_name)
        logger.info("✅ Pinecone initialized successfully")
    else:
        logger.warning("⚠️ PINECONE_API_KEY not found, will use mock data")
except Exception as e:
    logger.warning(f"⚠️ Pinecone initialization failed: {str(e)}")
    logger.warning("⚠️ Will use mock data instead")

# List all indexes to test the connection
try:
    if pc:
        indexes = pc.list_indexes()
        if indexes:
            logger.info(f"Connection successful! Available indexes: {indexes}")
        else:
            logger.warning("Connection successful but no indexes found.")
    else:
        logger.warning("Pinecone not initialized, cannot list indexes")
except Exception as e:
    logger.error(f"Failed to connect to Pinecone: {str(e)}")

# Load a pre-trained model and tokenizer
# Can test with gpt-40 or t5-large or another model to compare performance
try:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
except Exception as e:
    raise RuntimeError(f"Failed to load BERT model and tokenizer: {str(e)}")

def text_to_vector(text):
    try:
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        vector = torch.mean(outputs.last_hidden_state, dim=1).squeeze().tolist()
        return vector
    except Exception as e:
        logger.warning(f"Failed to convert text to vector: {str(e)}")
        return [0] * 768  # Return zero vector as fallback

def upsert_data_to_pinecone(data):
    try:
        if not data:
            raise ValueError("No data provided for upserting")
        vectors = [(f'vec_{i}', text_to_vector(item), {'source': 'firecrawl'})
                  for i, item in enumerate(data)]
        index.upsert(vectors)
    except Exception as e:
        raise RuntimeError(f"Failed to upsert data to Pinecone: {str(e)}")

def query_pinecone(query_text):
    """Query Pinecone with fallback to mock data"""
    try:
        if not query_text:
            raise ValueError("Query text cannot be empty")
        if pc and index:
            query_vector = text_to_vector(query_text)
            results = index.query(query_vector, top_k=5, include_metadata=True)
            return results
        else:
            raise RuntimeError("Pinecone not initialized")
    except Exception as e:
        logger.warning(f"Failed to query Pinecone: {str(e)}")
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
