# NBA Betting ML

Full-stack machine learning project predicting NBA game outcomes using
data from the NBA API and Basketball Reference.

## Stack
- **Data**: Python, nba_api, BeautifulSoup, pandas, scikit-learn
- **Database**: TBD
- **Backend**: Node.js / Express
- **Frontend**: Next.js / TypeScript / Tailwind CSS

## Structure
\```
nba-betting/
├── frontend/    # Next.js app (TypeScript)
├── api/         # Express REST API (Node.js)
├── python/      # Data scrapers + ML models
└── database/    # Schema files (coming soon)
\```

## Getting started

**Install dependencies:**
\```bash
cd api && npm install && cd ../frontend && npm install
cd ../python && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
\```

**Run frontend + API together:**
\```bash
npm run dev
\```

**Run Python scrapers:**
\```bash
cd python && source .venv/bin/activate
python3 nba/fetch_games.py
python3 scrapers/basketball_ref.py
\```

## Progress
- [ ] NBA API scraper
- [ ] Basketball Reference scraper
- [ ] Choose database
- [ ] Connect data layer to DB
- [ ] Feature engineering
- [ ] ML model v1
- [ ] Express API routes wired to DB
- [ ] Frontend predictions dashboard
