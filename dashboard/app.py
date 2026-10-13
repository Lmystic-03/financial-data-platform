from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "dashboard_data"
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha import capm_regression
from beta import rolling_beta
from portfolio import min_variance_weights, portfolio_vol

st.set_page_config(page_title="Financial Data Platform", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


st.title("Financial Data Platform")
st.caption("8 actions US + SPY, fenêtre glissante de 10 ans")

prices = load("prices_all")
tickers = sorted(prices["ticker"].unique())
stocks = [ticker for ticker in tickers if ticker != "SPY"]
returns = (
    prices.pivot(index="Date", columns="ticker", values="log_return")
    .dropna()
    .sort_index()
)

overview_tab, risk_tab, beta_alpha_tab, portfolio_tab, garch_tab = st.tabs(
    ["Vue d'ensemble", "Risque", "Bêta & alpha", "Portefeuille", "GARCH"]
)

with overview_tab:
    corr = load("correlation")
    st.header("Vue d'ensemble")

    chosen = st.multiselect("Tickers", tickers, default=["AAPL", "JPM", "SPY"])

    # Prix normalisés à 100 au premier jour, pour comparer des actions de niveaux différents.
    pivot = prices.pivot(index="Date", columns="ticker", values="Close")
    if chosen:
        norm = pivot[chosen] / pivot[chosen].iloc[0] * 100
        st.plotly_chart(
            px.line(norm, labels={"value": "Base 100", "Date": ""}),
            width="stretch",
        )
    else:
        st.info("Choisis au moins un ticker.")

    st.subheader("Corrélations des rendements")
    st.plotly_chart(
        px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        ),
        width="stretch",
    )

with risk_tab:
    st.header("Risque")

    risk = load("risk")  # index = ticker, valeurs en décimales.
    level = st.radio("Niveau de confiance", ["95", "99"], horizontal=True)

    cols = {
        f"hist_var_{level}": "VaR historique",
        f"param_var_{level}": "VaR paramétrique",
        f"es_{level}": "Expected Shortfall",
    }
    risk_view = (risk[list(cols)] * 100).rename(columns=cols)

    st.plotly_chart(
        px.bar(
            risk_view,
            barmode="group",
            labels={"value": "Perte journalière (%)", "index": ""},
        ),
        width="stretch",
    )
    st.dataframe(risk_view.round(2), width="stretch")
    st.caption(
        "Perte journalière maximale attendue (VaR) et perte moyenne au-delà (ES), "
        "en % du portefeuille."
    )

with beta_alpha_tab:
    st.header("Bêta glissant et alpha")

    ticker = st.selectbox("Ticker pour le bêta glissant (fenêtre de 250 jours)", stocks)
    market = returns["SPY"]
    beta = rolling_beta(returns[ticker], market).dropna().rename("Bêta glissant")
    st.plotly_chart(
        px.line(beta, labels={"value": "Bêta vs SPY", "Date": ""}),
        width="stretch",
    )

    st.subheader("Alpha du modèle de marché")
    alpha = pd.DataFrame(
        {stock: capm_regression(returns[stock], market) for stock in stocks}
    ).T
    alpha = alpha.rename(
        columns={
            "alpha_daily_%": "Alpha quotidien (%)",
            "alpha_annual_%": "Alpha annualisé (%)",
            "alpha_pvalue": "p-value alpha",
            "beta": "Bêta",
            "beta_pvalue": "p-value bêta",
            "r_squared": "R²",
        }
    )
    st.dataframe(alpha.round(4), width="stretch")

with portfolio_tab:
    st.header("Portefeuille")

    stock_returns = returns[stocks]
    equal_weights = pd.Series(1 / len(stocks), index=stocks)
    minimum_weights = min_variance_weights(stock_returns)
    equal_volatility = portfolio_vol(stock_returns, equal_weights)
    minimum_volatility = portfolio_vol(stock_returns, minimum_weights)

    equal_metric, minimum_metric = st.columns(2)
    equal_metric.metric("Volatilité équipondérée (annualisée)", f"{equal_volatility:.2%}")
    minimum_metric.metric(
        "Volatilité variance minimale (annualisée)", f"{minimum_volatility:.2%}"
    )

    weights = pd.DataFrame(
        {
            "Équipondéré": equal_weights,
            "Variance minimale": minimum_weights,
        }
    ) * 100
    st.plotly_chart(
        px.bar(
            weights,
            barmode="group",
            labels={"value": "Allocation (%)", "index": "Ticker"},
        ),
        width="stretch",
    )
    st.dataframe(weights.round(2), width="stretch")

with garch_tab:
    st.header("Volatilité GARCH(1,1)")

    garch_path = DATA / "garch_volatility.parquet"
    if garch_path.exists():
        garch = load("garch_volatility")
        garch = garch.rename(
            columns={"garch": "GARCH(1,1)", "rolling_20": "Volatilité glissante (20 j)"}
        )
        latest = garch["GARCH(1,1)"].iloc[-1]
        correlation = garch.corr().iloc[0, 1]

        latest_metric, correlation_metric = st.columns(2)
        latest_metric.metric("Volatilité GARCH actuelle (annualisée)", f"{latest:.2f} %")
        correlation_metric.metric("Corrélation avec la volatilité 20 j", f"{correlation:.3f}")

        st.plotly_chart(
            px.line(
                garch,
                labels={"value": "Volatilité annualisée (%)", "Date": "Date"},
            ),
            width="stretch",
        )
        st.caption("Les volatilités sont annualisées et exprimées en pourcentage.")
    else:
        st.warning("Le fichier de volatilité GARCH n'a pas encore été généré.")
        st.code("python src\\pipeline.py\npython src\\garch.py", language="powershell")
