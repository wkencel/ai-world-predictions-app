import openai
import instructor
import json
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Type, Union
from openai import OpenAI
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from termcolor import colored
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, Field
from models import DataPoints
from scraperUtils import extract_data_from_content, filter_empty_fields, create_filtered_model
from sportsLinks import extract_yahoo_sports_links

# Get fresh links from RSS feed
sports_links = extract_yahoo_sports_links()
print('sports_links: ', sports_links)
print('total sports_links: ', len(sports_links))

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client wrapped with Langsmith eventually for tracing
# this will require API keys and some research, but may prove useful
client = wrap_openai(openai.Client())

# Initialize Instructor client for handling tools (not used yet)
instructor_client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)

# Constants
GPT_MODEL = "gpt-4o"
MAX_TOKEN = 100000 # this can be adjusted

# Initialize the FirecrawlApp with your API key
app = FirecrawlApp()

# Function to save JSON array to a file in a pretty-printed format
def save_json_pretty(data, filename):
    """
    Append data to an existing JSON file or create a new one if it doesn't exist.
    
    Args:
        data: The new data to append (can be Pydantic model or dict)
        filename (str): The name of the file
    """
    try:
        # Convert Pydantic model to dict if necessary
        if hasattr(data, 'model_dump'):  # For Pydantic v2
            data = data.model_dump()
        elif hasattr(data, 'dict'):      # For Pydantic v1
            data = data.dict()
            
        # Initialize existing_data as an empty list
        existing_data = []
        
        # Try to read existing file
        try:
            with open(filename, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except FileNotFoundError:
            print(f"Creating new file: {filename}")
        except json.JSONDecodeError:
            print(f"Error reading existing file. Creating new file: {filename}")
            
        # Append new data
        if isinstance(data, list):
            existing_data.extend(data)
        else:
            existing_data.append(data)
            
        # Write back to file
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=4, sort_keys=True, ensure_ascii=False)
        print(f"Data successfully appended to {filename}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# careful with this, as scraping this many links could reach rate limits / cost $$
# if len(sports_links) > 60:
for link in sports_links[:2]:
    crawled_data = app.crawl_url(
        url=link,
        params={
            'limit': 1,
            'includePaths': [ r'-\d+\.html$' ],
            'ignoreSitemap': True,
            'scrapeOptions': {
                'formats': [ 'markdown' ],
                'excludeTags': [ '#ybar', '#sports-module-scorestrip', "#ad", ".advertisement", ".sponsored-content", ".link-yahoo-link", ".caas-img-container", ".caas-img-lightbox", ".link "],
                'includeTags': ["div.content", "h1", "p"],
                "onlyMainContent": True,
                "waitFor": 3000  # wait for 3 seconds for pages to load
            }
    })

    print('crawled_data total: ', crawled_data['total'])
    print('crawled_data status: ', crawled_data['status'])
    print('crawled_data creditsUsed: ', crawled_data['creditsUsed'])

    print('crawled_data for one entry json: ', crawled_data["data"][0])

# Sports Data Schema (these models will be moved out to a models directory)
    class Player(BaseModel):
        name: str = Field(..., description="The name of the player")
        team: str = Field(..., description="The team of the player")
        position: str = Field(..., description="The position of the player. Example: Forward, Center, etc.")
        stats: List[str] = Field(..., description="The stats of the player. If these are not available, you can add relevant stats from reliable sources like ESPN, Yahoo Sports, etc.")
        injuryStatus: str = Field(..., description="The injury status of the player")

    class Team(BaseModel):
        name: str = Field(..., description="The name of the team")
        people: List[Player] = Field(..., description="The players, coaches, and other relevant people in the team or mentioned in the article.")
        stats: List[str] = Field(..., description="The stats of the team. If these are not available, you can add relevant stats from reliable sources like ESPN, Yahoo Sports, etc.")

    class SportsArticleExtraction(BaseModel):
        title: str = Field(..., description="The title of the article")
        author: str = Field(..., description="The author of the article")
        date: str = Field(..., description="The date of the article")
        url: str = Field(..., description="The URL of the article")
        summary: str = Field(..., description="A detailed and descriptive summary of the article. It should not generalize facts or statistics, but rather provide a detailed summary of the article.")
        quote: str = Field(..., description="A quote from the article")
        players: List[Player] = Field(..., description="The players, coaches, and other relevant people in the article.")
        teams: List[Team] = Field(..., description="The teams in the article")

    # def extract_sports_data(crawled_data):
    for item in crawled_data["data"]:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at structured data extraction. You will be given unstructured markdown text from a website that contains articles related to sports news. You must extract the most important pieces of data that are relevant to the article. The data should be in the form of a JSON object that matches the SportsArticleExtraction schema as indicated in your instructions."},
                {"role": "user", "content": "Here is the markdown text for one article: " + item["markdown"][: (MAX_TOKEN)]}
            ],
            response_format=SportsArticleExtraction,
        )

        sports_data_json = completion.choices[0].message.parsed
        save_json_pretty(sports_data_json, "sports_data.json")

# print('sports_data_json: ', sports_data_json)

# save_json_pretty(crawled_data["data"][0]["json"], "crawled_data.json")

# data = app.scrape_url('https://docs.firecrawl.dev/', {
#     'formats': ['json'],
#     'jsonOptions': {
#         'schema': ExtractSchema.model_json_schema(),
#     }
# })