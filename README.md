# AI World Predictions App

The AI World Predictions App is designed to provide AI-driven predictions for world events using a Retrieval-Augmented Generation (RAG) approach. The application scrapes data from various websites, stores it in a vector database (Pinecone), and interacts with APIs such as Polymarket and Kalshi. This data is processed by an OpenAI language model and presented to users through a React-based web application.

## System Architecture

The application is divided into two main components: the client and the server.

### Client

The client is a React application that serves as the user interface for interacting with the AI predictions.

- **Public Directory**: Contains static assets.
  - `index.html`: The main HTML file for the React app.

- **Source Directory**: Contains the core application logic.
  - **components/**: Houses reusable React components.
  - **pages/**: Contains Next.js pages for routing and navigation.
  - **styles/**: Includes CSS/SCSS files for styling the application.
  - **hooks/**: Custom React hooks for managing state and side effects.
  - **utils/**: Utility functions for client-side operations.
  - **context/**: Context API files for global state management.
  - `App.js`: The main component that initializes the React app.
  - `index.js`: The entry point for the React application.

- `package.json`: Lists client-side dependencies and scripts.

### Server

The server handles data processing, API interactions, and serves the client application.

- **API Directory**: Manages external API integrations.
  - **polymarket/**: Handles interactions with the Polymarket API.
  - **kalshi/**: Manages connections to the Kalshi API.

- **Database Directory**: Manages database interactions.
  - **pinecone/**: Contains scripts and configurations for Pinecone vector database.

- **webScraping Directory**: Contains web scraping logic.
  - **models/**: Pydantic Data models for structuring data for json outputs
  - **crawlers/**: Scripts for scraping data by topic from websites.
  - **freshRSS/**: Scripts for extracting url links by topic from RSS feeds.
  - **outputs/**: Structured json outputs by topic after processing data froms scraping
  - **utils/**: Utility functions to support scraping operations

- **Services Directory**: Integrates with external services.
  - **openai/**: Manages interactions with OpenAI's language models.

- **Utilities Directory**: General utility functions for server-side logic.

- **Middlewares Directory**: Middleware functions for request processing.

- **Routes Directory**: Defines API endpoints and routing logic.

- **Models Directory**: Data models for structuring and validating data.

- **Controllers Directory**: Business logic and data handling.

- `server.js`: The main server file that initializes and runs the server.

- `package.json`: Lists server-side dependencies and scripts.

### Root

- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: Documentation for the project.

## Getting Started

To get started with the project, clone the repository and install the necessary dependencies for both the client and server. Then, run the development servers.

## Setup

1. **Run the Setup Script**:
   ```bash
   python setup.py
   ```
   This script will:
   - Create a Python virtual environment
   - Install Python dependencies
   - Install client-side npm dependencies

2. **Activate the Virtual Environment**:
   - On macOS and Linux:
     ```bash
     source ai-predict/bin/activate
     ```
   - On Windows:
     ```bash
     ai-predict\Scripts\activate
     ```

3. **Create a `.env` file** in the root directory (or in the `server` and/or `client` directories if you have separate configurations). This file will store your environment variables. For example:
   ```plaintext
   # Server environment variables
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   POLYMARKET_API_KEY=your_polymarket_api_key
   KALSHI_API_KEY=your_kalshi_api_key

   # Client environment variables
   REACT_APP_API_URL=your_api_url
   ```

   **Note**: Do not commit the `.env` file to git. It contains sensitive information like API keys and database credentials.

## Collaborators

- Ashley Pean
- Dave LaFlam
- Jamie Highsmith
- John Maltese
- Will Kencel

## Setting Up and Running the Pinecone Database

To use the Pinecone vector database with this application, follow these steps:

1. **Create an Index on Pinecone**:
   - Go to the [Pinecone website](https://www.pinecone.io/).
   - Log in or create an account if you don't have one.
   - Navigate to the "Indexes" section and click on "Create Index".
   - Choose the configuration "text-embedding-ada-002" for your index.

2. **Configure Your Environment**:
   - Ensure you have your Pinecone API key. You can find this in the Pinecone dashboard under the "API Keys" section.
   - Add your Pinecone API key to the `.env` file in the root directory:
     ```plaintext
     PINECONE_API_KEY=your_pinecone_api_key
     ```

3. **Run the Pinecone Setup Script**:
   - Navigate to the directory containing the `setup_pinecone.py` script:
     ```bash
     cd server/src/db/pinecone
     ```
   - Run the script to initialize and connect to your Pinecone index:
     ```bash
     python3 setup_pinecone.py
     ```

This will set up the connection to your Pinecone index and allow you to start using it with the application.

## Running the Web Scraping Service and Loading Data into Pinecone

### About the Web Scraping Service

**Required API Keys**
- OpenAI API Key formatted as constant `OPENAI_API_KEY`
- FireCrawl API Key formatted as constant `FIRECRAWL_API_KEY`
  - Note: FireCrawl is a paid service, but we have a free tier that offers 500 credits. However, 1 credit = 1 page so it goes quickly. Also, different paid tiers offer increasinly higher rate limits for number of crawl requests per minute. 

This service went through a few iterations, and the current iteration is the simplest solution and delivers the cleanest outputs. Here's a quick breakdown of the process:

1. **RSS Feeds**: 
  - The webscraping service currently relies on yahoo based sites, because they are easy to scrape and don't require any authentication.
  - The scripts in the freshRSS directory are designed to extract url links by their designated topic from RSS feeds. It parses the XML for <links> and then stores those links in a list. The process removes any duplicate links. 
  - The links are then sent to the crawlers to be scraped.

2. **Crawlers**:
  - The crawlers are designed to scrape the data from the websites and save it to a json file in the outputs directory. This is a two step process for each link.
  - The first step is to scrape the data from the website. This is done by leveraging the Firecrawl crawler endpoint to parse the website and return clean markdown text of the main article content.
  - The second step is to parse the data from the website. This is done by leveraging the OpenAI API. 
    - The AI model is instructed to take the markdown text and extract the most important pieces of data that are relevant to the article.
    - The AI model is instructed to output the data in the form of a JSON object that matches the specific schema for the topic (ex: SportsArticleExtraction).

3. **Loading Data into Pinecone**:
  - The crawlers use the jsonOutputParser utility to generate a pre-defined output file if it doesn't already exist. Otherwise, it will append fresh data to the existing file.
  *Need to Define Next Steps*
    - How are json files deleted once they are loaded into Pinecone?
    - How are the specific crawlers called and json data fetched for input into the Pinecone loader?
    - When/how does that json data go through an embedding process?
    - Are these all done manually once or twice a day for now? Will they ever need to be triggered by a user input (this process will take a few minutes to complete)?

To run the web scraping service and load data into Pinecone, follow these steps:

1. **Run the Web Scraping Service**:
   - Navigate to ai-world-predictions-app/server/src
   - Run the script for the specific topic you want to scrape (ex: sportsCrawler.py):
     ```bash
     python -m services.webScraping.crawlers.sportsCrawler 
     ```
     *Note: this might be python3 for you

    - This will create a json file in the outputs directory with the scraped data, labeled with the approriate name by topic. (Ex: sports_data.json)

2. **Load Data into Pinecone**:
    - These json files can then be loaded into Pinecone by running the following command:
    *NOTE:* We need to figure out the exact flow for this, I suggest from the db directory:
    - Calling the desired crawler script (or running a file that calls each of them one by one, to scrape and parse the data)
    - Then calling the load_pinecone.py script to load the data into Pinecone
    - After the data is successfully loaded into Pinecone, the script could delete the json file from the outputs directory (it will be re-created on the next run)

