### Scans FinViz.com news table for articles and derives the sentiment score for each ticker passed.

**SPY included as a baseline in all results.**

usage: `finviz-sentiment-analysis.py [-h] [-t TICKERS]`

i.e. `python finviz-sentiment-analysis.py -t AAPL -t MSFT -t META`

returns: `results=[('SPY', 0.7570469379425049), ('AAPL', 0.7831562757492065), ('MSFT',
0.8052746653556824), ('META', 0.8157443404197693)]`
