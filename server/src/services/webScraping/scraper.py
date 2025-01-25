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
from models import DataPoints
from scraperUtils import extract_data_from_content, filter_empty_fields, create_filtered_model

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

# Web scraping function
@traceable(run_type="tool", name="Scrape")
def scrape(url, data_points, links_scraped):
    """
    Scrape a given URL and extract structured data.
    
    Args:
    url (str): The URL to scrape.
    data_points (List[Dict]): The list of data points to extract.
    links_scraped (List[str]): List of already scraped links.
    
    Returns:
    dict: The extracted structured data or an error message.
    """
    app = FirecrawlApp()

    try:
        scraped_data = app.scrape_url(url)
        markdown = scraped_data["markdown"][: (MAX_TOKEN * 2)]
        links_scraped.append(url)

        extracted_data = extract_data_from_content(markdown, data_points, links_scraped, url)

        return extracted_data
    
    except Exception as e:
        print("Unable to scrape the url")
        print(f"Exception: {e}")
        return "Unable to scrape the url"
    

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
def call_agent(
    prompt, system_prompt, tools, plan, data_points, entity_name, links_scraped
):
    """
    Call the AI agent to perform tasks based on the given prompt and tools.

    Args:
        prompt (str): The user's prompt.
        system_prompt (str): The system instructions for the AI.
        tools (List[Dict]): Available tools for the AI to use.
        plan (bool): Whether to create a plan before execution.
        data_points (List[Dict]): The list of data points to extract.
        entity_name (str): The name of the entity being researched.
        links_scraped (List[str]): List of already scraped links.

    Returns:
        str: The final response from the AI agent.
    """
    messages = []

    if plan:
        messages.append(
            {
                "role": "user",
                "content": (
                    system_prompt
                    + "  "
                    + prompt
                    + "  Let's think step by step, make a plan first"
                ),
            }
        )

        chat_response = chat_completion_request(
            messages, tool_choice="none", tools=tools
        )
        messages = [
            {"role": "user", "content": (system_prompt + "  " + prompt)},
            {"role": "assistant", "content": chat_response.choices[0].message.content},
        ]

    else:
        messages.append({"role": "user", "content": (system_prompt + "  " + prompt)})

    state = "running"

    for message in messages:
        pretty_print_conversation(message)

    while state == "running":
        chat_response = chat_completion_request(messages, tool_choice=None, tools=tools)

        if isinstance(chat_response, Exception):
            print("Failed to get a valid response:", chat_response)
            state = "finished"
        else:
            current_choice = chat_response.choices[0]
            messages.append(
                {
                    "role": "assistant",
                    "content": current_choice.message.content,
                    "tool_calls": current_choice.message.tool_calls,
                }
            )
            pretty_print_conversation(messages[-1])
            
            if current_choice.finish_reason == "tool_calls":
                tool_calls = current_choice.message.tool_calls
                for tool_call in tool_calls:
                    function = tool_call.function.name
                    arguments = json.loads(
                        tool_call.function.arguments
                    )  # Parse the JSON string to a Python dict

                    if function == "scrape":
                        result = tools_list[function](
                            arguments["url"], data_points, links_scraped
                        )
                    elif function == "update_data":
                        result = tools_list[function](
                            data_points, arguments["datas_update"]
                        )
                    # elif function == "search":
                    #     result = tools_list[function](
                    #         arguments["query"], entity_name, data_points
                    #     )
                    # elif function == "file_reader":
                    #     result = tools_list[function](arguments["file_url"], links_scraped)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function,
                            "content": result,
                        }
                    )
                    pretty_print_conversation(messages[-1])

            if current_choice.finish_reason == "stop":
                state = "finished"

            # messages = memory_optimise(messages)

    return messages[-1]["content"]


# run agent to do website search
@traceable(name="#1 Website domain research")
def website_search(entity_name: str, website: str, data_points, links_scraped):
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

        response = call_agent(
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
def run_research(entity_name, website, data_points):
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

    response1 = website_search(entity_name, website, data_points, links_scraped)
    # response2 = internet_search(entity_name, website, data_points, links_scraped)

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

# Test Run
entity_name = "The Sporting News"
website = "https://www.sportingnews.com/us/soccer"
special_instruction = "This is a website of sports news, you should scrape the website to find the latest news about various sports. Look for as much data as possible, but don't scrape the same url twice. There are many links to articles as well as categories for articles broken down by sport. They are generally broken down by sport. You should scrape each sport section to get comprehensive  coverage of articles from each sport."


data_keys = list(DataPoints.__fields__.keys())
data_fields = DataPoints.__fields__

data_points = [{"name": key, "value": None, "reference": None, "description": data_fields[key].description} for key in data_keys]

data = run_research(entity_name, website, data_points)

# Specify the filename
filename = f"{entity_name}.json"

# Save the data
save_json_pretty(data_points, filename)

