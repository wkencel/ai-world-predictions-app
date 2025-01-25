# Web Scraping

This folder contains code for scraping websites and conducting web searches. 
This approach is not based on traditional libraries like BeautifulSoup or Scrapy, but rather uses the FireCrawl API and leverages LLMs, which has several advantages:
    - It requires less customization and knowledge of the underlying html structure of the website
    - It can scrape more data in a single request, which is useful for scraping large websites or websites with a lot of dynamic content
    - It can eventually leverage tools like AgentQL to conduct more complex scraping tasks, like login and bot detection
    - It can output human readable markdown, which can be used to update data in the database as well as JSON as structured output

This approach does rely on several other librairies:
    - `firecrawl` for calling the FireCrawl API (scraping/crawling)
    - `openai` for calling the OpenAI API (LLM)
    - `pydantic` for data modeling, validation, and parsing
    - `instructor` for calling the Instructor API (structured output, tools like scraping, searching, etc)
    - `langsmith` for tracing and debugging
    - `termcolor` for colored terminal output

## Required API Keys

- OpenAI API Key formatted as constant `OPENAI_API_KEY`
- FireCrawl API Key formatted as constant `FIRECRAWL_API_KEY`

## Scraping

- `scraper.py` is a basic scraper that scrapes a single url and updates data.
    - it can be run with `python scraper.py` from the webScraping folder in the server directory
    - it will generate a json file with the scraped data under the webScraping folder named after the entity_name as assigned in scraper.py

## Next Steps

- Add a function to scrape a list of urls and update data in the database
- Create asyncronous batch scraping 
- Employ the crawler function from firecrawl to thoroughly crawl websites, ensuring comprehensive data extraction while bypassing any web blocker mechanisms.
- Test scraping of social media starting with twitter






