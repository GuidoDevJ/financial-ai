"""Scraping fallback for outlets that publish no RSS feed. Isolated stub: NOT wired in.

The spec allows a light scrape of an outlet's "Mercados" section only when it has no RSS,
and only respecting its terms of use. Nothing here touches a live site.

TODO: before implementing, review each outlet's terms of use and robots.txt, rate-limit
requests, and only then parse the section page with BeautifulSoup into `NewsItem`s.
"""

from app.rag.ingest_news import NewsItem


def scrape_section(url: str) -> list[NewsItem]:
    raise NotImplementedError(
        "HTML scraping fallback is intentionally not implemented; see module docstring."
    )
