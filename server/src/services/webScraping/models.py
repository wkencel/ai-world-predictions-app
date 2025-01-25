from typing import List, Optional
from pydantic import BaseModel, Field

# Define the pydantic data structure for the data points
# These are specific to The Sporting News, but can be modified to fit other sports websites

class Player(BaseModel):
    name: str = Field(..., description="The name of the player")
    team: str = Field(..., description="The team of the player")
    position: str = Field(..., description="The position of the player")
    stats: List[str] = Field(..., description="The stats of the player")
    injuryStatus: str = Field(..., description="The injury status of the player")

class Team(BaseModel):
    name: str = Field(..., description="The name of the team")
    players: List[Player] = Field(..., description="The players in the team")
    stats: List[str] = Field(..., description="The stats of the team")

class NewsDataPoint(BaseModel):
    title: str = Field(..., description="The title of the article")
    url: str = Field(..., description="The URL of the article")
    content: str = Field(..., description="The content of the article")

class SocialDataPoint(BaseModel):
    platform: str = Field(..., description="The social media platform")
    content: str = Field(..., description="The content of the post")
    engagement: Optional[dict] = Field(None, description="Engagement metrics")

class DataPoints(BaseModel):
    """Base model for scraped data points"""
    title: str = Field(..., description="The title of the article")
    author: str = Field(..., description="The author of the article")
    date: str = Field(..., description="The date of the article")
    url: str = Field(..., description="The URL of the article")
    content: str = Field(..., description="The content of the article")
    quote: str = Field(..., description="A quote from the article")
    players: List[Player] = Field(..., description="The players in the article")
    teams: List[Team] = Field(..., description="The teams in the article")
