# WhatItMeans

A free, auto-updating site that explains what financial news **means** - not
just what happened.

Every story is broken down the same way:

**what happened → who is affected → which sectors → the transmission
mechanism → market reaction → bull case → bear case → variables to watch**

India-first, with global macro that transmits into India.

---

## What it does

- Groups duplicate/related coverage of the same story together
- Picks the most significant, well-corroborated stories
- Explains each one in plain language — mechanism, not just headline
- Shows both the bullish and bearish read on every story, clearly attributed
- Refreshes automatically around the clock, no manual updates needed
- 100% free to read, no login, no paywall, no tracking

## Why

Most financial news tells you *what* happened. Very little tells you *why it
matters* or *who it affects* — without slipping into "buy this, sell that"
territory. WhatItMeans is built to explain, not recommend.

> **Disclaimer:** WhatItMeans is an explanatory news product, not investment advice.
> Nothing on this site is a recommendation to buy, sell, or hold any security.
> Always do your own research or consult a licensed financial advisor before
> making investment decisions.

## Running it locally

```bash
pip install -r requirements.txt

# Add at least one LLM API key (see .env.example)
cp .env.example .env

# Build the site locally without publishing
python -m pipeline.run

# Preview
python -m http.server 8000 --directory site
```

Then open `http://localhost:8000`.

## Tests

```bash
python tests/test_compliance.py
python tests/test_router.py
python tests/test_render.py
python tests/test_ledger.py
```

## Tech

Python pipeline, static HTML output, deployed via GitHub Pages and GitHub
Actions. No database, no server to maintain.

## License

All rights reserved unless noted otherwise.
