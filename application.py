"""
application.py
Ayco AI Wealth Advisor — Flask Backend

Three integrated modules (mirrors Assignment 5 + NY Utility deployment pattern):
  1. RAG Financial Research Assistant  — retrieval-augmented AI chatbot
  2. Black-Scholes Options Pricer      — classic quant finance model
  3. Portfolio Risk Dashboard          — VaR + Sharpe + Volatility

Architecture:
  - Pickle model loaded at startup (same as utility_rates.pkl pattern)
  - Flask routes serve HTML templates
  - Black-Scholes computed in pure Python (no external dependencies)
  - RAG uses keyword retrieval over pickled knowledge base
"""

from flask import Flask, request, render_template, jsonify
import pickle, math, json
from statistics import mean, stdev

app = Flask(__name__)

# ── Load pickled financial model (mirrors utility_rates.pkl pattern) ──────────
model         = pickle.load(open("financial_model.pkl", "rb"))
BENCHMARKS    = model["benchmarks"]
RF_RATE       = model["risk_free_rate"]
SECTORS       = model["sectors"]
RISK_PROFILES = model["risk_profiles"]
RAG_KB        = model["rag_kb"]

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — BLACK-SCHOLES OPTIONS PRICER
# ═══════════════════════════════════════════════════════════════════════════════

def normal_cdf(x):
    """Cumulative standard normal distribution (Abramowitz & Stegun approximation)."""
    a1, a2, a3, a4, a5 = 0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    L = abs(x)
    k = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-L * L / 2) * \
        (a1*k + a2*k**2 + a3*k**3 + a4*k**4 + a5*k**5)
    return w if x >= 0 else 1.0 - w

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes European option price.
    S: current stock price
    K: strike price
    T: time to expiry (years)
    r: risk-free rate (decimal)
    sigma: implied volatility (decimal)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None, None, None

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        price = S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
        delta = normal_cdf(d1)
    else:
        price = K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)
        delta = normal_cdf(d1) - 1

    # Greeks
    gamma = math.exp(-d1**2 / 2) / (S * sigma * math.sqrt(T) * math.sqrt(2 * math.pi))
    theta = (-(S * sigma * math.exp(-d1**2 / 2)) / (2 * math.sqrt(T) * math.sqrt(2 * math.pi))
             - r * K * math.exp(-r * T) * (normal_cdf(d2) if option_type == "call" else normal_cdf(-d2))) / 365
    vega  = S * math.sqrt(T) * math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi) * 0.01

    return round(price, 4), round(delta, 4), round(gamma, 6), round(theta, 4), round(vega, 4), round(d1, 4), round(d2, 4)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — PORTFOLIO RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def compute_portfolio_risk(holdings, portfolio_value):
    """
    Compute VaR, Sharpe Ratio, and Volatility for a user portfolio.
    holdings: dict of {asset_key: weight (0-1)}
    portfolio_value: total $ value
    """
    # Weighted return and volatility
    weighted_return = 0
    weighted_variance = 0

    for asset, weight in holdings.items():
        if asset in BENCHMARKS:
            b = BENCHMARKS[asset]
            weighted_return   += weight * b["avg_annual_return"]
            weighted_variance += (weight * b["std_dev"]) ** 2

    portfolio_return = weighted_return
    portfolio_std    = math.sqrt(weighted_variance)  # simplified (assumes 0 correlation)

    # Sharpe Ratio
    sharpe = (portfolio_return - RF_RATE) / portfolio_std if portfolio_std > 0 else 0

    # VaR (95% confidence, 30-day)
    z_95  = 1.645
    var_30d = portfolio_value * z_95 * portfolio_std * math.sqrt(30/252)

    # Volatility label
    if portfolio_std < 0.08:
        vol_label, vol_class = "Low", "low"
    elif portfolio_std < 0.16:
        vol_label, vol_class = "Medium", "medium"
    else:
        vol_label, vol_class = "High", "high"

    # Sharpe label
    if sharpe < 0:
        sharpe_label, sharpe_class = "Poor", "poor"
    elif sharpe < 1.0:
        sharpe_label, sharpe_class = "Below Average", "below"
    elif sharpe < 2.0:
        sharpe_label, sharpe_class = "Good", "good"
    else:
        sharpe_label, sharpe_class = "Excellent", "excellent"

    # Best matching risk profile
    best_profile = min(RISK_PROFILES.keys(),
                       key=lambda p: abs(RISK_PROFILES[p]["target_return"] - portfolio_return))

    return {
        "portfolio_return": f"{portfolio_return:.2%}",
        "portfolio_std":    f"{portfolio_std:.2%}",
        "sharpe":           f"{sharpe:.2f}",
        "sharpe_label":     sharpe_label,
        "sharpe_class":     sharpe_class,
        "var_30d":          f"${var_30d:,.0f}",
        "var_30d_pct":      f"{(var_30d/portfolio_value)*100:.1f}%",
        "vol_label":        vol_label,
        "vol_class":        vol_class,
        "risk_profile":     RISK_PROFILES[best_profile]["label"],
        "risk_description": RISK_PROFILES[best_profile]["description"],
        "rf_rate":          f"{RF_RATE:.2%}",
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — RAG FINANCIAL RESEARCH ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════

def rag_retrieve(query, top_k=3):
    """
    Keyword-based retrieval from the financial knowledge base.
    Returns top_k most relevant entries based on term overlap.
    """
    query_terms = set(query.lower().split())
    scored = []

    for entry in RAG_KB:
        text    = (entry["topic"] + " " + entry["content"] + " " + entry["category"]).lower()
        overlap = sum(1 for t in query_terms if t in text)
        # Boost for topic title match
        if any(t in entry["topic"].lower() for t in query_terms):
            overlap += 3
        scored.append((overlap, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k] if _ > 0]

def rag_answer(query):
    """
    Generate a structured answer using retrieved context from the knowledge base.
    """
    docs = rag_retrieve(query, top_k=3)

    if not docs:
        return {
            "answer": "I couldn't find specific information about that topic in my knowledge base. Try asking about: Black-Scholes, Sharpe Ratio, VaR, S&P 500, interest rates, inflation, asset allocation, bonds, or portfolio theory.",
            "sources": [],
            "query": query
        }

    # Build context-grounded answer
    context_parts = []
    for doc in docs:
        context_parts.append(f"[{doc['topic']}] {doc['content']}")

    primary = docs[0]

    # Rule-based response generation grounded in retrieved content
    answer = f"**{primary['topic']}**\n\n{primary['content']}"

    if len(docs) > 1:
        answer += f"\n\n**Related: {docs[1]['topic']}**\n{docs[1]['content']}"

    return {
        "answer":  answer,
        "sources": [{"topic": d["topic"], "category": d["category"]} for d in docs],
        "query":   query
    }

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

# ── Black-Scholes ──────────────────────────────────────────────────────────────
@app.route("/options")
def options_page():
    return render_template("options.html", rf_rate=RF_RATE)

@app.route("/options/price", methods=["POST"])
def price_option():
    try:
        S           = float(request.form.get("stock_price", 0))
        K           = float(request.form.get("strike_price", 0))
        T           = float(request.form.get("days_to_expiry", 0)) / 365
        sigma       = float(request.form.get("volatility", 0)) / 100
        r           = float(request.form.get("risk_free", RF_RATE * 100)) / 100
        option_type = request.form.get("option_type", "call")

        result = black_scholes(S, K, T, r, sigma, option_type)
        if result[0] is None:
            return render_template("options.html", rf_rate=RF_RATE,
                                   error="Invalid inputs. Check that all values are positive.")

        price, delta, gamma, theta, vega, d1, d2 = result
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        time_value = round(price - intrinsic, 4)

        return render_template("options.html",
            rf_rate=RF_RATE, S=S, K=K, T_days=int(T*365), sigma_pct=sigma*100,
            r_pct=r*100, option_type=option_type,
            price=price, delta=delta, gamma=gamma, theta=theta, vega=vega,
            d1=d1, d2=d2, intrinsic=intrinsic, time_value=time_value,
            moneyness="In the Money" if intrinsic > 0 else ("At the Money" if intrinsic == 0 else "Out of the Money"),
            moneyness_class="itm" if intrinsic > 0 else ("atm" if intrinsic == 0 else "otm"),
        )
    except (ValueError, TypeError) as e:
        return render_template("options.html", rf_rate=RF_RATE, error=f"Input error: {str(e)}")

# ── Risk Dashboard ─────────────────────────────────────────────────────────────
@app.route("/risk")
def risk_page():
    assets = {k: v["name"] for k, v in BENCHMARKS.items()}
    return render_template("risk.html", assets=assets)

@app.route("/risk/analyze", methods=["POST"])
def analyze_risk():
    try:
        portfolio_value = float(request.form.get("portfolio_value", 0))
        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be positive")

        # Parse holdings weights
        holdings = {}
        total_weight = 0
        for asset in BENCHMARKS.keys():
            w = float(request.form.get(f"weight_{asset}", 0))
            if w > 0:
                holdings[asset] = w / 100
                total_weight += w

        if total_weight == 0:
            raise ValueError("Please enter at least one asset allocation")

        # Normalize weights to 100%
        holdings = {k: v / (total_weight / 100) for k, v in holdings.items()}

        result = compute_portfolio_risk(holdings, portfolio_value)

        # Build holdings display
        holdings_display = [
            {"name": BENCHMARKS[k]["name"], "weight": f"{v*100:.1f}%",
             "ret": f"{BENCHMARKS[k]['avg_annual_return']:.2%}",
             "vol": f"{BENCHMARKS[k]['std_dev']:.2%}"}
            for k, v in holdings.items()
        ]

        assets = {k: v["name"] for k, v in BENCHMARKS.items()}
        return render_template("risk.html", assets=assets,
                               portfolio_value=f"${portfolio_value:,.0f}",
                               holdings=holdings_display,
                               **result)
    except (ValueError, TypeError) as e:
        assets = {k: v["name"] for k, v in BENCHMARKS.items()}
        return render_template("risk.html", assets=assets, error=str(e))

# ── RAG Chatbot ────────────────────────────────────────────────────────────────
@app.route("/research")
def research_page():
    sample_questions = [
        "What is the Black-Scholes formula?",
        "How do I calculate Sharpe Ratio?",
        "What is Value at Risk (VaR)?",
        "Explain Modern Portfolio Theory",
        "What does the Federal Reserve do to markets?",
        "What is bond duration?",
        "How does inflation affect investments?",
        "What is the VIX fear index?",
    ]
    return render_template("research.html", sample_questions=sample_questions)

@app.route("/research/ask", methods=["POST"])
def ask_research():
    query = request.form.get("query", "").strip()
    if not query:
        sample_questions = [
            "What is the Black-Scholes formula?",
            "How do I calculate Sharpe Ratio?",
            "What is Value at Risk (VaR)?",
            "Explain Modern Portfolio Theory",
        ]
        return render_template("research.html", sample_questions=sample_questions,
                               error="Please enter a question.")

    result = rag_answer(query)
    sample_questions = [
        "What is the Black-Scholes formula?",
        "How do I calculate Sharpe Ratio?",
        "What is Value at Risk (VaR)?",
        "Explain Modern Portfolio Theory",
    ]
    return render_template("research.html", sample_questions=sample_questions,
                           query=result["query"], answer=result["answer"],
                           sources=result["sources"])

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8888)
