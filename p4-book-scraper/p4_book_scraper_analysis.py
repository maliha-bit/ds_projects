"""
Python Project A: Web scraper + data analysis pipeline.
Scrapes books.toscrape.com (a public practice site built for scraping), cleans the data
with pandas, and analyses price vs rating.

    python p4_book_scraper_analysis.py            # scrape 10 pages (200 books)
    python p4_book_scraper_analysis.py --pages 50 # all 1000 books

Outputs: books.csv + outputs_p4/ charts and summary.
"""
import argparse
import os
import re
import time

import matplotlib
import pandas as pd
import requests
from bs4 import BeautifulSoup

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "https://books.toscrape.com/catalogue/page-{}.html"
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
OUT = "outputs_p4"


def scrape_page(page):
    r = requests.get(BASE.format(page), timeout=15)
    r.encoding = "utf-8"
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    for card in soup.select("article.product_pod"):
        rating_word = card.select_one("p.star-rating")["class"][1]
        price_text = card.select_one("p.price_color").text
        rows.append(
            {
                "title": card.h3.a["title"],
                "price": float(re.search(r"[\d.]+", price_text).group()),
                "rating": RATING_MAP.get(rating_word),
                "in_stock": "In stock" in card.select_one("p.availability").text,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    data = []
    for p in range(1, args.pages + 1):
        try:
            data += scrape_page(p)
            print(f"Scraped page {p}/{args.pages} ({len(data)} books)")
        except requests.RequestException as e:
            print(f"Page {p} failed: {e}")
        time.sleep(0.5)  # be polite to the server

    df = pd.DataFrame(data).drop_duplicates("title")
    df.to_csv("books.csv", index=False)

    summary = df.groupby("rating")["price"].agg(["count", "mean", "median"]).round(2)
    print("\nPrice by rating:\n", summary)
    summary.to_csv(f"{OUT}/price_by_rating.csv")

    plt.figure(figsize=(6, 4))
    df["price"].hist(bins=30)
    plt.title("Book price distribution (GBP)")
    plt.xlabel("Price")
    plt.tight_layout()
    plt.savefig(f"{OUT}/price_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    summary["mean"].plot(kind="bar")
    plt.title("Average price by star rating")
    plt.ylabel("Mean price")
    plt.tight_layout()
    plt.savefig(f"{OUT}/price_by_rating.png", dpi=150)
    plt.close()

    print(f"\nSaved books.csv and charts in ./{OUT}/")


if __name__ == "__main__":
    main()
