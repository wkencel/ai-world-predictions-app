from pinecone import Pinecone
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import logging

# Add at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Determine path to .env file
env_path = os.path.join(os.path.dirname(__file__), '../../../../.env')

# Verify the path exists before loading
if not os.path.exists(env_path):
    raise FileNotFoundError(f".env file not found at {env_path}")

load_dotenv(env_path)


pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise EnvironmentError("PINECONE_API_KEY not found in environment variables")


# Initialize Pinecone with error handling
try:
    pc = Pinecone(api_key=pinecone_api_key)
except Exception as e:
    raise ConnectionError(f"Failed to initialize Pinecone client: {str(e)}")

# Use the existing index
index_name = "ai-world-predictions-pinecone"

# Connect to the index with error handling
try:
    index = pc.Index(index_name)
except Exception as e:
    raise ConnectionError(f"Failed to connect to Pinecone index '{index_name}': {str(e)}")

# List all indexes to test the connection
try:
    indexes = pc.list_indexes()
    if indexes:
        logger.info(f"Connection successful! Available indexes: {indexes}")
    else:
        logger.warning("Connection successful but no indexes found.")
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
        raise ValueError(f"Failed to convert text to vector: {str(e)}")

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
    try:
        if not query_text:
            raise ValueError("Query text cannot be empty")
        query_vector = text_to_vector(query_text)
        results = index.query(query_vector, top_k=5, include_metadata=True)
        return results
    except Exception as e:
        raise RuntimeError(f"Failed to query Pinecone: {str(e)}")