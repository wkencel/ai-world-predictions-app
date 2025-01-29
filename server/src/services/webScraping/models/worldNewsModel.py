from typing import List
from pydantic import BaseModel, Field

# Define the pydantic data structure for the data points
# These are specific to US News

# US News Data Schema
class Figure(BaseModel):
    name: str = Field(..., description="The name of the person")
    organization: str = Field(..., description="The organization, company, political party, institution, teamn, etc that the person is associated with")
    position: str = Field(..., description="The position of the person. Example: CEO, President, etc.")
    facts: List[str] = Field(..., description="The facts about the person. If these are not available, you can add relevant facts from reliable sources like the BBC, Reuters, Bloomberg, CNN, AP, etc.")

class Organization(BaseModel):
    name: str = Field(..., description="The name of the organization")
    people: List[Figure] = Field(..., description="The figureheads, donors, funders, sponsors, leaders, workers, and other relevant people in the organization or mentioned in the article.")
    facts: List[str] = Field(..., description="The facts about the organization. If these are not available, you can add relevant facts from reliable sources like Reuters, Bloomberg, CNN, AP, etc.")

class WorldNewsArticleExtraction(BaseModel):
    title: str = Field(..., description="The title of the article")
    author: str = Field(..., description="The author of the article")
    date: str = Field(..., description="The date of the article")
    url: str = Field(..., description="The URL of the article")
    summary: str = Field(..., description="A detailed and descriptive summary of the article. It should not generalize facts or statistics, but rather provide a thorough, specific, and detailed summary of the article.")
    quote: str = Field(..., description="A quote from the article")
    figures: List[Figure] = Field(..., description="The figureheads, donors, funders, sponsors, leaders, workers, and other relevant people in the article.")
    organizations: List[Organization] = Field(..., description="The organizations, companies, institutions, teams, and other relevant organizations in the article.")