import sys
import torch
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as soup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from rich import print

def get_page(url):
    request = Request(url=url, headers={'user-agent': 'Not-A-Browser'})
    response = urlopen(request)
    return soup(response, "lxml")

def estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]

finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ["SPY"]
portfolio = {}
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"{device=}")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]


def main():
    for t in tickers:
        portfolio[t] = {
            "sentiment" : 1,
            "links" : {}
        }

    for ticker in tickers:
        url = finviz_url + ticker
        html = get_page(url)
        news = html.find(id="news-table").findAll('tr')
        for index, row in enumerate(news):
            if row.a and index < 6:
                title = row.a.get_text()
                link = row.a.get("href")
                portfolio[ticker]["links"][title] = {}
                portfolio[ticker]["links"][title]["url"] = link

    try:
        for t in tqdm(portfolio, desc="portfolio"):
            try:
                f = portfolio[t]
                for v in tqdm(f["links"].values(), desc=t):
                    try:
                        url = v['url']
                        count_prob = []
                        results = get_page(url).findAll('p')
                        for p in results:
                            try:
                                news = p.get_text()
                                probability, sentiment = estimate_sentiment(news)
                                count_prob.append(probability)
                            except RuntimeError:
                                continue
                            except KeyboardInterrupt:
                                sys.exit()
                    except KeyboardInterrupt:
                        sys.exit()
            except KeyboardInterrupt:
                sys.exit()
            portfolio[t]["sentiment"] = (sum(count_prob) / len(count_prob)).item()
    except KeyboardInterrupt:
        sys.exit()

    return [(p, portfolio[p]["sentiment"]) for p in portfolio]

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="""
Scans FinViz.com news table for articles and derives the sentiment score for each ticker passed.
SPY included as a baseline in all results.""")
    parser.add_argument("-t", "--tickers", action="append", type=str, help="pass a string of tickers separated by spaces. SPY included by default.")
    args = parser.parse_args()
    if args.tickers:
        tickers += args.tickers

    results = main()
    print(f"{results=}")