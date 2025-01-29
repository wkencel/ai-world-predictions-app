import openai
import instructor
import json
import time  # Add this import at the top with your other imports
from firecrawl import FirecrawlApp
from typing import List, Optional, Dict, Any, Type, Union
from openai import OpenAI
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langsmith.wrappers import wrap_openai
import os

from ..freshRSS.sportsLinks import extract_yahoo_sports_links
from ..models.sportsModel import SportsArticleExtraction
from ..utils.jsonOutputParser import save_json_pretty

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

# Start timer for first loop
start_time_crawl = time.time()

# careful with this, as scraping this many links could reach rate limits / cost $$
# if len(sports_links) > 60:
for link in sports_links[:2]: # adjust this to scrape more links
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
    print('crawled_data for one entry: ', crawled_data["data"][0])

    # End timer for first loop and print duration
    end_time_crawl = time.time()
    print(f"URL crawling took {end_time_crawl - start_time_crawl:.2f} seconds")

    # Start timer for second loop
    start_time_process = time.time()

    for item in crawled_data["data"]:
        # ! NICK FUN TIMES
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

    # End timer for second loop and print duration
    end_time_process = time.time()
    print(f"Data processing to JSON took {end_time_process - start_time_process:.2f} seconds")
