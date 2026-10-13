import re
import time

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

from kgrag.config import BASE_URL, SCHEME_LIST_URL

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0"
session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2)))


def _soup(url: str) -> BeautifulSoup:
    time.sleep(1)  # be polite to the server
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def scheme_links() -> list[str]:
    soup = _soup(SCHEME_LIST_URL)
    hrefs = {a["href"] for a in soup.select('a[href*="scheme_details.php?id="]')}
    return sorted(BASE_URL + h.lstrip("/") for h in hrefs)


def scheme_details(url: str) -> dict:
    """Parse the label/value table of a scheme page into a dict."""
    fields = {"url": url}
    for row in _soup(url).select("tr"):
        cells = row.find_all("td")
        if len(cells) != 2 or not cells[0].find("b"):
            continue
        label = cells[0].get_text(strip=True).rstrip(":").strip()
        value = re.sub(r"\s*\n\s*", "\n", cells[1].get_text("\n", strip=True)).strip()
        value = value.replace("`", "Rs.")  # the site renders the rupee symbol as a backtick
        if value and value.lower() not in {"-", "na", "n/a", "nil"}:
            fields[label] = value
    return fields
