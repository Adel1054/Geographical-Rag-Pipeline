import os
import requests
from bs4 import BeautifulSoup
import time


URLS = {
    "Land": "https://www.britannica.com/place/France/Land",
    "Hercynian_massifs": "https://www.britannica.com/place/France/The-Hercynian-massifs",
    "Great_lowlands": "https://www.britannica.com/place/France/The-great-lowlands",
    "Younger_mountains": "https://www.britannica.com/place/France/The-younger-mountains-and-adjacent-plains",
    "Drainage": "https://www.britannica.com/place/France/Drainage",
    "Soils": "https://www.britannica.com/place/France/Soils",
    "Climate": "https://www.britannica.com/place/France/Climate",
    "Plant_and_animal_life": "https://www.britannica.com/place/France/Plant-and-animal-life"
}

SAVE_DIR = "../data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}
session = requests.Session()
session.headers.update(headers)


for title, url in URLS.items():
    print(f"Scraping {title}...")
    try:
        # Use session and add a 10-second timeout
        res = session.get(url, timeout=10)
        res.raise_for_status()  # Raises an exception for HTTP errors (like 404 or 500)

        soup = BeautifulSoup(res.content, "html.parser")

        article = soup.find("article")
        if not article:
            print(f"Article tag not found for {title}")
            continue

        paragraphs = article.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs])

        with open(os.path.join(SAVE_DIR, f"{title}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved {title}.txt")

        # Politeness delay of 2 seconds
        time.sleep(2)

    except requests.exceptions.RequestException as e:
        print(f"Network error while scraping {title}: {e}")
    except Exception as e:
        print(f"Failed to parse {title}: {e}")