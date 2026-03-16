# Ayco AI Wealth Advisor
### A Quantitative Finance Portfolio Project — Goldman Sachs Ayco

A full-stack Flask application combining three financial intelligence modules:
RAG-powered research, Black-Scholes options pricing, and institutional portfolio risk analysis.
Built on the same ML deployment architecture as DTSC-680 Assignment 5.

---

## Three Modules

| Module | Description |
|---|---|
| 🤖 RAG Research Assistant | Retrieval-augmented Q&A over a curated financial knowledge base |
| 📈 Black-Scholes Options Pricer | European call/put pricing with full Greeks (Δ, Γ, Θ, V) |
| ⚠️ Portfolio Risk Dashboard | VaR (95%), Sharpe Ratio, volatility classification |

---

## How to Run Locally

### 1. Install dependencies
```bash
pip install flask gunicorn
```

### 2. Build the financial model
```bash
python build_financial_model.py
```
Output: `✓ Financial model built: 5 benchmarks, 10 sector profiles, 15 RAG entries`

### 3. Run the app
```bash
python application.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

---

## Project Structure

```
ayco_advisor/
├── application.py            # Flask backend — all three module routes
├── build_financial_model.py  # Builds and pickles the financial data model
├── financial_model.pkl       # Pickled model (benchmarks, RAG KB, risk profiles)
├── requirements.txt
├── README.md
└── templates/
    ├── base.html             # Shared layout, nav, design system
    ├── index.html            # Overview / landing page
    ├── research.html         # RAG research assistant
    ├── options.html          # Black-Scholes options pricer
    ├── risk.html             # Portfolio risk dashboard
    └── about.html            # Architecture & data sources
```

---

## Architecture

Mirrors the DTSC-680 Assignment 5 Flask deployment pattern:
- Financial data model pickled at build time (`build_financial_model.py`)
- Loaded at runtime by `application.py` (same as `utility_rates.pkl` / `titanic.pkl`)
- Three route modules serve HTML templates via Jinja2
- Pure-Python Black-Scholes with no external math dependencies
- RAG retrieval using keyword scoring + topic boosting over pickled knowledge base

---

## Technical Highlights

- **Black-Scholes**: Implemented from scratch using Abramowitz & Stegun normal CDF approximation
- **VaR**: Parametric, 95% confidence, 30-day horizon: `VaR = PV × 1.645 × σ × √(30/252)`
- **Sharpe Ratio**: `(Rp - Rf) / σp` against 2024 10Y Treasury rate of 4.25%
- **RAG**: Keyword overlap scoring with topic-title boost, top-k retrieval, grounded generation

---

## Disclaimer

For educational/portfolio demonstration purposes only. Not affiliated with Goldman Sachs or Ayco.
Data is representative and should not be used for investment decisions.
