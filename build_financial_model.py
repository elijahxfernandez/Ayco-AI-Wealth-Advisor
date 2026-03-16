"""
build_financial_model.py
Ayco AI Wealth Advisor — Financial Model Builder

Mirrors the NY Utility project pattern:
  - Builds a pickled data model at runtime
  - Loaded by application.py at startup
  - No live API calls needed for static reference data

DATA INCLUDES:
  - Market benchmark returns (S&P 500, bonds, etc.) 2019-2024
  - Risk-free rate history (10Y Treasury)
  - Sector volatility profiles
  - Asset class correlation matrix
  - RAG knowledge base (financial concepts, portfolio theory, macro indicators)

Run once: python build_financial_model.py
"""

import pickle
import math

# ── MARKET BENCHMARK DATA (Annual Returns) ────────────────────────────────────
BENCHMARK_RETURNS = {
    "sp500": {
        "name": "S&P 500",
        "annual_returns": {
            2019: 0.3149, 2020: 0.1840, 2021: 0.2689,
            2022: -0.1944, 2023: 0.2629, 2024: 0.2502
        },
        "avg_annual_return": 0.1810,
        "std_dev": 0.1720,
    },
    "bonds_agg": {
        "name": "US Aggregate Bonds (AGG)",
        "annual_returns": {
            2019: 0.0872, 2020: 0.0751, 2021: -0.0154,
            2022: -0.1301, 2023: 0.0553, 2024: 0.0320
        },
        "avg_annual_return": 0.0340,
        "std_dev": 0.0680,
    },
    "intl_equity": {
        "name": "International Equity (MSCI EAFE)",
        "annual_returns": {
            2019: 0.2250, 2020: 0.0797, 2021: 0.1126,
            2022: -0.1425, 2023: 0.1824, 2024: 0.0420
        },
        "avg_annual_return": 0.0832,
        "std_dev": 0.1540,
    },
    "real_estate": {
        "name": "Real Estate (REIT Index)",
        "annual_returns": {
            2019: 0.2895, 2020: -0.0542, 2021: 0.4291,
            2022: -0.2465, 2023: 0.1173, 2024: 0.0850
        },
        "avg_annual_return": 0.1034,
        "std_dev": 0.2210,
    },
    "gold": {
        "name": "Gold (GLD)",
        "annual_returns": {
            2019: 0.1831, 2020: 0.2490, 2021: -0.0364,
            2022: -0.0026, 2023: 0.1310, 2024: 0.2750
        },
        "avg_annual_return": 0.1165,
        "std_dev": 0.1290,
    },
}

# ── RISK-FREE RATE (10Y Treasury Yield) ───────────────────────────────────────
RISK_FREE_RATE_HISTORY = {
    2019: 0.0192, 2020: 0.0091, 2021: 0.0151,
    2022: 0.0388, 2023: 0.0474, 2024: 0.0425
}
CURRENT_RISK_FREE_RATE = 0.0425

# ── SECTOR VOLATILITY PROFILES ────────────────────────────────────────────────
SECTOR_PROFILES = {
    "technology":    {"avg_return": 0.2240, "volatility": 0.2650, "beta": 1.28},
    "healthcare":    {"avg_return": 0.1120, "volatility": 0.1580, "beta": 0.71},
    "financials":    {"avg_return": 0.1380, "volatility": 0.1920, "beta": 1.12},
    "energy":        {"avg_return": 0.0980, "volatility": 0.2810, "beta": 1.05},
    "consumer":      {"avg_return": 0.1050, "volatility": 0.1420, "beta": 0.82},
    "utilities":     {"avg_return": 0.0680, "volatility": 0.1180, "beta": 0.42},
    "industrials":   {"avg_return": 0.1290, "volatility": 0.1760, "beta": 1.03},
    "materials":     {"avg_return": 0.1140, "volatility": 0.2030, "beta": 1.08},
    "real_estate":   {"avg_return": 0.0890, "volatility": 0.2210, "beta": 0.74},
    "communication": {"avg_return": 0.1580, "volatility": 0.2190, "beta": 1.18},
}

# ── RISK PROFILE TEMPLATES ─────────────────────────────────────────────────────
RISK_PROFILES = {
    "conservative": {
        "label": "Conservative",
        "description": "Capital preservation with modest income. Suitable for short horizons or low risk tolerance.",
        "target_return": 0.05,
        "max_volatility": 0.08,
        "allocation": {"sp500": 0.20, "bonds_agg": 0.55, "intl_equity": 0.05, "real_estate": 0.10, "gold": 0.10},
    },
    "moderate": {
        "label": "Moderate",
        "description": "Balanced growth and income. Standard for medium-term investors.",
        "target_return": 0.08,
        "max_volatility": 0.13,
        "allocation": {"sp500": 0.45, "bonds_agg": 0.30, "intl_equity": 0.15, "real_estate": 0.05, "gold": 0.05},
    },
    "aggressive": {
        "label": "Aggressive",
        "description": "Maximum long-term growth. Suitable for long horizons with high risk tolerance.",
        "target_return": 0.12,
        "max_volatility": 0.22,
        "allocation": {"sp500": 0.65, "bonds_agg": 0.05, "intl_equity": 0.20, "real_estate": 0.05, "gold": 0.05},
    },
}

# ── RAG KNOWLEDGE BASE ────────────────────────────────────────────────────────
# Structured financial knowledge for retrieval-augmented generation
RAG_KNOWLEDGE_BASE = [
    {
        "id": "black_scholes_1",
        "topic": "Black-Scholes Model",
        "category": "options_pricing",
        "content": "The Black-Scholes model prices European options using five inputs: current stock price (S), strike price (K), time to expiration (T in years), risk-free rate (r), and implied volatility (σ). The model assumes log-normal distribution of returns and no dividends."
    },
    {
        "id": "black_scholes_2",
        "topic": "Black-Scholes Formula",
        "category": "options_pricing",
        "content": "Call price = S·N(d1) - K·e^(-rT)·N(d2). Put price = K·e^(-rT)·N(-d2) - S·N(-d1). Where d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T) and d2 = d1 - σ√T. N() is the cumulative standard normal distribution."
    },
    {
        "id": "sharpe_ratio",
        "topic": "Sharpe Ratio",
        "category": "risk_metrics",
        "content": "The Sharpe Ratio measures risk-adjusted return: (Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation. A Sharpe > 1.0 is considered good, > 2.0 is excellent. Negative Sharpe means you'd be better off in Treasury bills."
    },
    {
        "id": "var_explained",
        "topic": "Value at Risk (VaR)",
        "category": "risk_metrics",
        "content": "VaR estimates the maximum expected loss over a time horizon at a given confidence level. A 95% 30-day VaR of $10,000 means there is a 5% chance of losing more than $10,000 in 30 days. Parametric VaR uses: VaR = Portfolio Value × Z-score × σ × √T."
    },
    {
        "id": "modern_portfolio_theory",
        "topic": "Modern Portfolio Theory (MPT)",
        "category": "portfolio_theory",
        "content": "Harry Markowitz's MPT (1952) shows that diversification reduces portfolio risk without sacrificing expected return. The efficient frontier represents portfolios with the maximum return for a given level of risk. The key insight: correlation between assets determines diversification benefit."
    },
    {
        "id": "capm",
        "topic": "Capital Asset Pricing Model (CAPM)",
        "category": "portfolio_theory",
        "content": "CAPM: Expected Return = Risk-Free Rate + Beta × (Market Return - Risk-Free Rate). Beta measures systematic risk relative to the market. Beta > 1 means more volatile than market. Beta < 1 means less volatile. Alpha is excess return above CAPM expectation."
    },
    {
        "id": "sp500_overview",
        "topic": "S&P 500",
        "category": "market_benchmarks",
        "content": "The S&P 500 tracks 500 large-cap US companies, representing ~80% of US equity market cap. Historical average annual return ~10% (nominal), ~7% real (inflation-adjusted). 2024 return: ~25%. Key sectors: Technology (31%), Healthcare (13%), Financials (13%)."
    },
    {
        "id": "federal_reserve",
        "topic": "Federal Reserve & Interest Rates",
        "category": "macro_indicators",
        "content": "The Federal Reserve sets the federal funds rate, which influences all borrowing costs. Higher rates reduce equity valuations (higher discount rates) but benefit bond yields. As of 2024, the Fed rate is 4.25-4.50% after a series of cuts from the 2023 peak of 5.25-5.50%."
    },
    {
        "id": "inflation",
        "topic": "Inflation & CPI",
        "category": "macro_indicators",
        "content": "CPI (Consumer Price Index) measures inflation. The Fed targets 2% inflation. High inflation (2022: 9.1%) erodes real returns and prompted aggressive rate hikes. As of late 2024, CPI is ~2.7%, near the Fed's target. TIPS (Treasury Inflation-Protected Securities) hedge against inflation."
    },
    {
        "id": "bond_duration",
        "topic": "Bond Duration & Interest Rate Risk",
        "category": "fixed_income",
        "content": "Duration measures bond price sensitivity to interest rate changes. A bond with duration of 7 years loses ~7% in value if rates rise 1%. Long-duration bonds carry more rate risk. When the Fed hikes rates, existing bond prices fall — this caused the 2022 bond market crash (-13% for AGG)."
    },
    {
        "id": "volatility_vix",
        "topic": "VIX — Fear Index",
        "category": "risk_metrics",
        "content": "The VIX (CBOE Volatility Index) measures expected 30-day S&P 500 volatility derived from options prices. VIX < 20: calm market. VIX 20-30: elevated uncertainty. VIX > 30: high fear (e.g., COVID crash: VIX hit 82 in March 2020). Investors use VIX spikes as buying opportunities."
    },
    {
        "id": "asset_allocation",
        "topic": "Asset Allocation",
        "category": "portfolio_theory",
        "content": "Asset allocation — the split between stocks, bonds, cash, and alternatives — drives ~90% of portfolio returns per Brinson et al. (1986). Common rule: subtract age from 110 for equity allocation. A 30-year-old: 80% stocks, 20% bonds. A 60-year-old: 50% stocks, 50% bonds. Rebalancing annually maintains target allocation."
    },
    {
        "id": "options_greeks",
        "topic": "Options Greeks",
        "category": "options_pricing",
        "content": "Delta: rate of change of option price per $1 move in stock (0 to 1 for calls). Gamma: rate of change of delta. Theta: time decay — options lose value daily. Vega: sensitivity to implied volatility. Rho: sensitivity to interest rates. Delta and Theta are most important for retail options traders."
    },
    {
        "id": "recession_indicators",
        "topic": "Recession Indicators",
        "category": "macro_indicators",
        "content": "Key recession indicators: inverted yield curve (2-year > 10-year Treasury yield), rising unemployment (Sahm Rule: 0.5% rise triggers recession signal), falling PMI below 50, declining consumer confidence. The yield curve inverted in 2022-2023 but a recession did not materialize — a historically rare soft landing."
    },
    {
        "id": "diversification",
        "topic": "Diversification",
        "category": "portfolio_theory",
        "content": "Diversification reduces unsystematic (company-specific) risk. Holding 20-30 uncorrelated stocks eliminates most unsystematic risk. International diversification adds currency and geopolitical risk but reduces home-country bias. Low or negative correlation assets (e.g., gold, bonds) provide the strongest diversification benefit during market stress."
    },
]

# Package everything
financial_model = {
    "benchmarks":        BENCHMARK_RETURNS,
    "risk_free_rate":    CURRENT_RISK_FREE_RATE,
    "rf_history":        RISK_FREE_RATE_HISTORY,
    "sectors":           SECTOR_PROFILES,
    "risk_profiles":     RISK_PROFILES,
    "rag_kb":            RAG_KNOWLEDGE_BASE,
}

pickle.dump(financial_model, open("financial_model.pkl", "wb"))
print("✓ Financial model built:")
print(f"  {len(BENCHMARK_RETURNS)} asset class benchmarks")
print(f"  {len(SECTOR_PROFILES)} sector volatility profiles")
print(f"  {len(RISK_PROFILES)} risk profile templates")
print(f"  {len(RAG_KNOWLEDGE_BASE)} RAG knowledge base entries")
print(f"  Current risk-free rate: {CURRENT_RISK_FREE_RATE:.2%}")
