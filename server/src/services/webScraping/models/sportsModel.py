from typing import List
from pydantic import BaseModel, Field

# Define the pydantic data structure for the data points
# These are specific to The Sporting News

# Sports Data Schema 
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
    summary: str = Field(..., description="A detailed and descriptive summary of the article. It should not generalize facts or statistics, but rather provide a thorough, specific, and detailed summary of the article.")
    quote: str = Field(..., description="A quote from the article")
    players: List[Player] = Field(..., description="The players, coaches, and other relevant people in the article.")
    teams: List[Team] = Field(..., description="The teams in the article")