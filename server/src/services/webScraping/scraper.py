import os
import re
import time
import tempfile
import sys
import requests
import openai
import instructor
import json
import tiktoken
from typing import List, Optional, Dict, Any, Type, Union
from openai import OpenAI
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from termcolor import colored
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, Field
from .models import DataPoints, Player, Team
from .scraperUtils import extract_data_from_content, filter_empty_fields, create_filtered_model
from utils.logger import color_logger
from datetime import datetime
from bs4 import BeautifulSoup

try:
    import instructor
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    color_logger.warning("Instructor module not available. Some features may be limited.")

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client wrapped with Langsmith eventually for tracing
# this will require API keys and some research, but may prove useful
client = wrap_openai(openai.Client())

# Initialize Instructor client for handling tools
# Currently scraping a specific site and updating data, later could load and search docs or conduct web search
instructor_client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)

# Constants
GPT_MODEL = "gpt-4o"
MAX_TOKEN = 100000 # this can be adjusted
# LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

@traceable(run_type="tool", name="Scrape URL")
async def scrape(url: str, data_points: List[Dict], links_scraped: List[str]) -> str:
    """
    Scrape content from a URL and extract structured data.

    Args:
        url: The URL to scrape
        data_points: List of data points to extract
        links_scraped: List of already scraped URLs
    """
    try:
        if url in links_scraped:
            return "URL already scraped"

        color_logger.info(f"🔍 Scraping URL: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text(separator=' ', strip=True)

        # Extract data using the existing utility
        extracted_data = extract_data_from_content(content, data_points, links_scraped, url)

        links_scraped.append(url)
        return json.dumps(extracted_data, indent=2)

    except Exception as e:
        color_logger.error(f"Error scraping {url}: {str(e)}")
        return f"Error: {str(e)}"

class WebScraper:
    """Web scraping component for the prediction framework"""

    def __init__(self):
        self.links_scraped = []
        self.MAX_TOKEN = 100000
        color_logger.info("🌐 WebScraper initialized")

    async def fetch_data(self) -> Dict:
        """Fetch data from configured web sources"""
        try:
            scraped_data = {
                'news': await self._scrape_news(),
                'social': await self._scrape_social(),
                'timestamp': datetime.now().isoformat()
            }
            color_logger.info("✅ Web scraping completed successfully")
            return scraped_data
        except Exception as e:
            color_logger.error(f"❌ Error in web scraping: {str(e)}")
            return {}

    async def _scrape_news(self) -> List[Dict]:
        """Scrape news sources"""
        try:
            color_logger.info("📰 Scraping news sources...")
            # Use the existing scrape function for news sources
            news_data = []
            for url in self.get_news_urls():
                result = await scrape(url, [], self.links_scraped)
                if isinstance(result, dict):
                    news_data.append(result)
            return news_data
        except Exception as e:
            color_logger.error(f"Error in news scraping: {str(e)}")
            return []

    async def _scrape_social(self) -> List[Dict]:
        """Scrape social media sources"""
        try:
            color_logger.info("🐦 Scraping social media...")
            # Use the existing scrape function for social media sources
            social_data = []
            for url in self.get_social_urls():
                result = await scrape(url, [], self.links_scraped)
                if isinstance(result, dict):
                    social_data.append(result)
            return social_data
        except Exception as e:
            color_logger.error(f"Error in social media scraping: {str(e)}")
            return []

    def get_news_urls(self) -> List[str]:
        """Get list of news URLs to scrape"""
        # Implement your news source URLs here
        return []

    def get_social_urls(self) -> List[str]:
        """Get list of social media URLs to scrape"""
        # Implement your social media source URLs here
        return []

    def process_data(self, data: Dict) -> Dict:
        """Process scraped data"""
        try:
            if not data:
                return {}

            processed_data = {
                'processed_news': self._process_news(data.get('news', [])),
                'processed_social': self._process_social(data.get('social', [])),
                'timestamp': datetime.now().isoformat()
            }
            color_logger.info("✅ Data processing completed")
            return processed_data
        except Exception as e:
            color_logger.error(f"❌ Error processing scraped data: {str(e)}")
            return {}

    def _process_news(self, news_data: List) -> List[Dict]:
        """Process news data"""
        return []

    def _process_social(self, social_data: List) -> List[Dict]:
        """Process social media data"""
        return []

# Build agent runtime
@traceable(run_type="tool", name="Update data points")
def update_data(data_points, datas_update):
    """
    Update the state with new data points found.

    Args:
        data_points (list): The current data points state
        datas_update (List[dict]): The new data points found, have to follow the format [{"name": "xxx", "value": "xxx", "reference": "xxx"}]

    Returns:
        str: A message indicating the update status
    """
    print(f"Updating the data {datas_update}")

    try:
        for data in datas_update:
            for obj in data_points:
                if obj["name"] == data["name"]:
                    if data["type"].lower() == "dict":
                        obj["reference"] = data["reference"] if data["reference"] else "None"
                        obj["value"] = json.loads(data["value"])
                    elif data["type"].lower() == "str":
                        obj["reference"] = data["reference"]
                        obj["value"] = data["value"]
                    elif data["type"].lower() == "list":
                        if isinstance(data["value"], str):
                            data_value = json.loads(data["value"])
                        else:
                            data_value = data["value"]

                        for item in data_value:
                            item["reference"] = data["reference"]

                        if obj["value"] is None:
                            obj["value"] = data_value
                        else:
                            obj["value"].extend(data_value)

        return "data updated"
    except Exception as e:
        print("Unable to update data points")
        print(f"Exception: {e}")
        return "Unable to update data points"

@traceable(run_type="llm", name="Agent chat completion")
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
def chat_completion_request(messages, tool_choice, tools, model=GPT_MODEL):
    """
    Make a chat completion request to the OpenAI API.

    Args:
        messages (List[Dict]): The conversation history.
        tool_choice (str): The chosen tool for the AI to use.
        tools (List[Dict]): Available tools for the AI to use.
        model (str): The GPT model to use.

    Returns:
        openai.ChatCompletion: The response from the OpenAI API.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def pretty_print_conversation(message):
    """
    Print a conversation message with color-coding based on the role.

    Args:
        message (Dict): The message to print.
    """
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }

    if message["role"] == "system":
        print(colored(f"system: {message['content']}", role_to_color[message["role"]]))
    elif message["role"] == "user":
        print(colored(f"user: {message['content']}", role_to_color[message["role"]]))
    elif message["role"] == "assistant" and message.get("tool_calls"):
        print(
            colored(
                f"assistant: {message['tool_calls']}\n",
                role_to_color[message["role"]],
            )
        )
    elif message["role"] == "assistant" and not message.get("tool_calls"):
        print(
            colored(
                f"assistant: {message['content']}\n", role_to_color[message["role"]]
            )
        )
    elif message["role"] == "tool":
        print(
            colored(
                f"function ({message['name']}): {message['content']}\n",
                role_to_color[message["role"]],
            )
        )

# Dictionary of available tools
tools_list = {
    "scrape": scrape,
    "update_data": update_data,
    # "search": search,
    # "file_reader": llama_parser,
}

@traceable(name="Optimise memory")
def memory_optimise(messages: list):
    """
    Optimize the conversation history to fit within token limits.

    Args:
        messages (List[Dict]): The full conversation history.

    Returns:
        List[Dict]: The optimized conversation history.
    """
    system_prompt = messages[0]["content"]

    # token count
    encoding = tiktoken.encoding_for_model(GPT_MODEL)

    if len(encoding.encode(str(messages))) > MAX_TOKEN:
        latest_messages = messages
        token_count_latest_messages = len(encoding.encode(str(latest_messages)))
        print(f"initial Token count of latest messages: {token_count_latest_messages}")

        while token_count_latest_messages > MAX_TOKEN:
            latest_messages.pop(0)
            token_count_latest_messages = len(encoding.encode(str(latest_messages)))
            print(f"Token count of latest messages: {token_count_latest_messages}")

        print(f"Final Token count of latest messages: {token_count_latest_messages}")

        index = messages.index(latest_messages[0])
        early_messages = messages[:index]

        prompt = f""" {early_messages}
        -----
        Above is the past history of conversation between user & AI, including actions AI has already taken
        Please summarize the past actions taken so far, specifically around:
        - What data sources has the AI looked up already
        - What data points have been found so far

        SUMMARY:
        """

        response = client.chat.completions.create(
            model=GPT_MODEL, messages=[{"role": "user", "content": prompt}]
        )

        system_prompt = f"""{system_prompt}; Here is a summary of past actions taken so far: {response.choices[0].message.content}"""
        messages = [{"role": "system", "content": system_prompt}] + latest_messages

        return messages

    return messages

@traceable(name="Call agent")
async def call_agent(prompt, system_prompt, tools, plan, data_points, entity_name, links_scraped):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    state = "running"
    while state == "running":
        try:
            chat_response = await chat_completion_request(messages, tool_choice=None, tools=tools)

            if isinstance(chat_response, Exception):
                raise chat_response

            current_choice = chat_response.choices[0]
            messages.append({
                "role": "assistant",
                "content": current_choice.message.content,
                "tool_calls": current_choice.message.tool_calls,
            })

            if current_choice.finish_reason == "tool_calls":
                tool_calls = current_choice.message.tool_calls
                for tool_call in tool_calls:
                    function = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    result = None
                    if function == "scrape":
                        result = await tools_list[function](arguments["url"], data_points, links_scraped)
                    elif function == "update_data":
                        result = await tools_list[function](data_points, arguments["datas_update"])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function,
                        "content": result,
                    })
            else:
                state = "finished"

        except Exception as e:
            color_logger.error(f"Error in call_agent: {str(e)}")
            state = "finished"

    return messages[-1].get("content", "Failed to generate response")

# run agent to do website search
@traceable(name="#1 Website domain research")
async def website_search(entity_name: str, website: str, data_points, links_scraped):
    """
    Perform a search on the entity's website to find relevant information.

    Args:
        entity_name (str): The name of the entity being researched.
        website (str): The website URL of the entity.
        data_points (List[Dict]): The list of data points to extract.
        links_scraped (List[str]): List of already scraped links.

    Returns:
        str: The response from the AI agent after searching the website.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "scrape",
                "description": "Scrape a URL for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "the url of the website to scrape",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_reader",
                "description": "Get content from a file url that ends with pdf or img extension, e.g. https://xxxxx.jpg",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_url": {
                            "type": "string",
                            "description": "the url of the pdf or image file",
                        }
                    },
                    "required": ["file_url"],
                },
            },
        }
    ]

    data_keys_to_search = [
        {"name": obj["name"], "description": obj["description"]}
        for obj in data_points
        if obj["value"] is None or obj["value"] == "None"
    ]

    if len(data_keys_to_search) > 0:
        system_prompt = f"""
        you are a world class web scraper, you are great at finding information on urls;
        You will scrape pages within a company/entity's domain to find specific data about the company/entity, but You NEVER make up links, ONLY scrape real links you found or given

        {special_instruction}

        You only use data retrieved from scraper, do not make things up;
        You DO NOT scrape the same url twice, if you already scraped a url, do not scrape it again;

        You NEVER ask user for inputs or permissions, just go ahead do the best thing possible without asking for permission or guidance;

        All result will be auto logged & saved, so your final output doesn't need to repeat info gathered, just output "All info found"
        """

        prompt = f"""
        Entity to search: {entity_name}

        Company Website: {website}

        Data points to find:
        {data_keys_to_search}
        """

        response = await call_agent(
            prompt,
            system_prompt,
            tools,
            plan=False,
            data_points=data_points,
            entity_name=entity_name,
            links_scraped=links_scraped,
        )

        return response

# optional: run agent to do internet search
# @traceable(name="#2 Internet search")
# def internet_search(entity_name: str, website: str, data_points, links_scraped):
#     """
#     Perform an internet search to find additional information about the entity.

#     Args:
#         entity_name (str): The name of the entity being researched.
#         website (str): The website URL of the entity.
#         data_points (List[Dict]): The list of data points to extract.
#         links_scraped (List[str]): List of already scraped links.

#     Returns:
#         str: The response from the AI agent after performing the internet search.
#     """
#     tools = [
#         {
#             "type": "function",
#             "function": {
#                 "name": "search",
#                 "description": "Search internet for information & related urls",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "query": {
#                             "type": "string",
#                             "description": "the search query, should be semantic search query, as we are using a very smart semantic search engine; but always ask direct question",
#                         },
#                         "entity_name": {
#                             "type": "string",
#                             "description": "the name of the entity that we are researching about",
#                         },
#                     },
#                     "required": ["query", "entity_name"],
#                 },
#             },
#         },
#     ]

#     data_keys_to_search = [
#         {"name": obj["name"], "description": obj["description"]}
#         for obj in data_points
#         if obj["value"] is None or obj["value"] == "None"
#     ]

#     if len(data_keys_to_search) > 0:
#         system_prompt = """
#         you are a world class web researcher
#         You will keep doing web search based on information you received until all information is found;

#         You will try as hard as possible to search for all sorts of different query & source to find information; if one search query didnt return any result, try another one;
#         You do not stop until all information are found, it is very important we find all information, I will give you $200,000 tip if you find all information;

#         You only answer questions based on results from scraper, do not make things up;
#         You never ask user for inputs or permissions, you just do your job and provide the results;
#         You ONLY run 1 function at a time, do NEVER run multiple functions at the same time

#         All result will be auto logged & saved, so your final output doesn't need to repeat info gathered, just output "All info found"
#         """

#         prompt = f"""
#         Entity to search: {entity_name}

#         Entity's website: {website}

#         Links we already scraped: {links_scraped}

#         Data points to find:
#         {data_keys_to_search}
#         """

#         response = call_agent(
#             prompt,
#             system_prompt,
#             tools,
#             plan=False,
#             data_points=data_points,
#             entity_name=entity_name,
#             links_scraped=links_scraped,
#         )

#         return response

@traceable(name="Run research")
async def run_research(entity_name, website, data_points):
    """
    Run the complete research process for an entity.

    Args:
        entity_name (str): The name of the entity being researched.
        website (str): The website URL of the entity.
        data_points (List[Dict]): The list of data points to extract.

    Returns:
        List[Dict]: The updated data points after research.
    """
    links_scraped = []

    response1 = await website_search(entity_name, website, data_points, links_scraped)
    # response2 = await internet_search(entity_name, website, data_points, links_scraped)

    return data_points

# Function to save JSON array to a file in a pretty-printed format
def save_json_pretty(data, filename):
    """
    Save a JSON array to a file in a pretty-printed format.

    Args:
        data: The data to be saved as JSON.
        filename (str): The name of the file to save the data to.
    """
    try:
        print(f"Saving data to {filename}")
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, sort_keys=True, ensure_ascii=False)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Test Run - Wrap in async function and use asyncio
async def main():
    entity_name = "The Sporting News"
    website = "https://www.sportingnews.com/us/soccer"
    special_instruction = "This is a website of sports news, you should scrape the website to find the latest news about various sports. Look for as much data as possible, but don't scrape the same url twice. There are many links to articles as well as categories for articles broken down by sport. They are generally broken down by sport. You should scrape each sport section to get comprehensive  coverage of articles from each sport."

    data_keys = list(DataPoints.__fields__.keys())
    data_fields = DataPoints.__fields__

    data_points = [{"name": key, "value": None, "reference": None, "description": data_fields[key].description} for key in data_keys]

    data = await run_research(entity_name, website, data_points)

    # Specify the filename
    filename = f"{entity_name}.json"

    # Save the data
    save_json_pretty(data_points, filename)

# Add this at the bottom of the file
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
