# Empty init file to make this a package

from .models import DataPoints
from .scraper import WebScraper

__all__ = ['DataPoints', 'WebScraper']
