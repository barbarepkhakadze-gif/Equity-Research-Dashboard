import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import io
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG + FONT
# -----------------------------
st.set_page_config(page_title="Equity Research Terminal", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: "Times New Roman", serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Equity Research Terminal")

# -----------------------------
# DATA ENGINE
# -----------------------------
def get_metrics(ticker):

    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")

    if hist.empty or len(hist) < 30:
        return None

    price = hist["Close"]

    r1 = ((price.iloc[-1] / price.iloc[-21]) - 1) * 100
    r12 = ((price.iloc[-1] / price.iloc[0]) - 1) * 100

    returns = price.pct_change().dropna()

    vol = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    score = (r12 * 0.4) + (r1 * 0.2) + (sharpe * 10) - (vol * 0.3)

    if r12 > 25:
        trend = "strong long-term appreciation"
    elif r12 > 10:
        trend = "moderate upward trend"
    else:
        trend = "weak or inconsistent long-term performance"

    if r1 > 5:
        short_term = "positive recent momentum"
    elif r1 > -5:
        short_term = "neutral short-term movement"
    else:
        short_term = "negative recent momentum"

    if sharpe > 1:
        efficiency = "strong risk-adjusted efficiency"
    elif sharpe > 0.5:
        efficiency = "moderate risk-adjusted efficiency"
    else:
        efficiency = "low risk-adjusted efficiency"

    if vol > 40:
        risk_view = "high volatility (elevated uncertainty)"
    elif vol > 20:
        risk_view = "moderate volatility"
    else:
        risk_view = "low volatility (stable price behavior)"

    if score > 20:
        signal = "BUY"
        verdict = "expected return outweighs risk profile"
    elif score > 5:
        signal = "HOLD"
        verdict = "balanced risk-reward with no strong edge"
    else:
        signal = "SELL"
        verdict = "risk-adjusted returns are insufficient"

    risk_flags = []
    if vol > 40:
        risk_flags.append("High Volatility")
    if sharpe < 0:
        risk_flags.append("Negative Sharpe Ratio")
    if r12 < 0:
        risk_flags.append("Negative Annual Return")

    reasoning = f"""
PRICE & RETURN PROFILE:
- Annual Return: {r12:.2f}%
- Volatility: {vol:.2f}%
- Sharpe: {sharpe:.2f}
- Trend: {trend}

RISK SUMMARY:
{", ".join(risk_flags) if risk_flags else "No major risks"}

FINAL VIEW:
{signal} — {verdict}
"""

    return {
        "Ticker": ticker,
        "Company": info.get("longName", ticker),
        "Price": price.iloc[-1],
        "Return": r12,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Signal": signal,
        "Reason": reasoning,
        "Risk Flags": risk_flags,
        "History": price
    }

# -----------------------------
# INPUT
# -----------------------------
tickers_input = st.text_input("Enter tickers", "AAPL,MSFT,NVDA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# -----------------------------
# RUN
# -----------------------------
if st.button("Run Analysis"):

    results = []
    price_data = {}

    for t in tickers:
        data = get_metrics(t)
        if data:
            results.append(data)
            price_data[data["Company"]] = data["History"]

    df = pd.DataFrame(results)

    st.session_state.df = df
    st.session_state.price_data = price_data

if "df" not in st.session_state:
    st.stop()

df = st.session_state.df
price_data = st.session_state.price_data

df = df.sort_values(by="Return", ascending=False).reset_index(drop=True)
df.index = df.index + 1

# -----------------------------
# OPTIMIZER (UNCHANGED)
# -----------------------------
def optimize_portfolio(returns):

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n = len(mean_returns)

    def neg_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return / portfolio_vol)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.array([1/n] * n)

    result = minimize(
        neg_sharpe,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x, mean_returns, cov_matrix

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Analysis",
    "Portfolio",
    "Risk",
    "Report",
    "Optimizer"
])

# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    st.dataframe(df, use_container_width=True)

# -----------------------------
# TAB 2
# -----------------------------
with tab2:
    selected = st.selectbox("Select stock", df["Company"])
    row = df[df["Company"] == selected].iloc[0]

    st.subheader(selected)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Return", f"{row['Return']:.2f}%")
    with col2:
        st.metric("Sharpe", round(row["Sharpe"], 2))
    with col3:
        st.metric("Signal", row["Signal"])

    history = row["History"].to_frame()
    history.columns = ["Price"]
    history = history.sort_index()

    st.line_chart(history, use_container_width=True)
    st.write(row["Reason"])

# -----------------------------
# TAB 3
# -----------------------------
with tab3:
    clean = {k: v.sort_index() for k, v in price_data.items()}
    st.line_chart(pd.DataFrame(clean), use_container_width=True)

# -----------------------------
# TAB 4
# -----------------------------
with tab4:
    returns = pd.DataFrame(price_data).pct_change().dropna()
    st.dataframe(returns.corr())

# -----------------------------
# TAB 5 (UPGRADED PDF)
# -----------------------------
with tab5:

    def create_pdf(df, price_data):

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        content = []

        content.append(Paragraph("EQUITY RESEARCH REPORT", styles["Title"]))
        content.append(Spacer(1, 12))

        # -----------------------------
        # STOCK TABLE
        # -----------------------------
        table_data = [["Stock", "Return", "Sharpe", "Signal"]]
        for _, row in df.iterrows():
            table_data.append([
                row["Company"],
                f"{row['Return']:.2f}%",
                f"{row['Sharpe']:.2f}",
                row["Signal"]
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.grey),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.5, colors.black)
        ]))

        content.append(table)
        content.append(Spacer(1, 20))

        # -----------------------------
        # PORTFOLIO OPTIMIZATION INSIDE PDF
        # -----------------------------
        returns = pd.DataFrame(price_data).pct_change().dropna()
        weights, mean_returns, cov = optimize_portfolio(returns)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = portfolio_return / portfolio_vol

        content.append(Paragraph("PORTFOLIO OPTIMIZATION", styles["Heading2"]))
        content.append(Paragraph(f"Expected Return: {portfolio_return*100:.2f}%", styles["Normal"]))
        content.append(Paragraph(f"Volatility: {portfolio_vol*100:.2f}%", styles["Normal"]))
        content.append(Paragraph(f"Sharpe Ratio: {sharpe:.2f}", styles["Normal"]))
        content.append(Spacer(1, 12))

        # weight table
        weight_table = [["Stock", "Weight"]]
        for i, col in enumerate(returns.columns):
            weight_table.append([col, f"{weights[i]*100:.2f}%"])

        wt = Table(weight_table)
        wt.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.black)
        ]))

        content.append(wt)
        content.append(Spacer(1, 20))

        # -----------------------------
        # PORTFOLIO GRAPH
        # -----------------------------
        fig, ax = plt.subplots()
        returns.cumsum().plot(ax=ax)
        ax.set_title("Cumulative Returns (Portfolio Assets)")

        img_path = "portfolio.png"
        plt.savefig(img_path)
        plt.close()

        content.append(Image(img_path, width=400, height=250))

        doc.build(content)
        buffer.seek(0)
        return buffer

    if st.button("Generate Report"):
        pdf = create_pdf(df, price_data)

        st.download_button(
            "Download Report",
            pdf,
            file_name="equity_research_report.pdf",
            mime="application/pdf"
        )

# -----------------------------
# TAB 6
# -----------------------------
with tab6:

    returns = pd.DataFrame(price_data).pct_change().dropna()

    weights, mean_returns, cov = optimize_portfolio(returns)

    portfolio_return = np.dot(weights, mean_returns)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe = portfolio_return / portfolio_vol

    result_df = pd.DataFrame({
        "Stock": returns.columns,
        "Weight": weights
    })

    st.dataframe(result_df)