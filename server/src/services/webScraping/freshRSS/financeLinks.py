import openai
import xml.etree.ElementTree as ET
import requests
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langsmith.wrappers import wrap_openai

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client wrapped with Langsmith eventually for tracing
# this will require API keys and some research, but may prove useful
client = wrap_openai(openai.Client())

# Constants
GPT_MODEL = "gpt-4o"
MAX_TOKEN = 100000 # this can be adjusted


# Initialize the FirecrawlApp with API key
app = FirecrawlApp()


def extract_finance_links():
    """
    Fetches and parses the Yahoo Finance RSS feed to extract unique links.
    
    Returns:
        list: A list of unique URLs found in the RSS feed
    """
    url = "https://finance.yahoo.com/news/rssindex"
    links_set = set()  # No duplicates
    
    try:
        # Fetch the RSS feed
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the XML content
        root = ET.fromstring(response.content)
        
        # Find all <link> elements and extract their text
        for link in root.findall('.//link'):
            if link.text and link.text.startswith('http'):  # Only include valid URLs
                links_set.add(link.text)
        
        # Convert set back to list for return
        return list(links_set)
        
    except requests.RequestException as e:
        print(f"Error fetching RSS feed: {e}")
        return []
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return []

# To run file directly:
if __name__ == "__main__":
    finance_links = extract_finance_links()
    print(f"Found {len(finance_links)} links:")
    for link in finance_links:
        print(link)