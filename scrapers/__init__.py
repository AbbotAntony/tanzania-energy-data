"""One module per data source. ``REGISTRY`` maps CLI names to scraper classes."""

from __future__ import annotations

from scrapers.base import Scraper
from scrapers.bot import BotScraper
from scrapers.ewura import EwuraScraper
from scrapers.news_citizen import CitizenScraper
from scrapers.news_dailynews import DailyNewsScraper
from scrapers.nbs import NbsScraper
from scrapers.nishati import NishatiScraper
from scrapers.tanesco import TanescoScraper
from scrapers.tpdc import TpdcScraper

REGISTRY: dict[str, type[Scraper]] = {
    EwuraScraper.name: EwuraScraper,
    BotScraper.name: BotScraper,
    TpdcScraper.name: TpdcScraper,
    NishatiScraper.name: NishatiScraper,
    TanescoScraper.name: TanescoScraper,
    DailyNewsScraper.name: DailyNewsScraper,
    CitizenScraper.name: CitizenScraper,
    NbsScraper.name: NbsScraper,
}
