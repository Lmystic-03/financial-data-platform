from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "dashboard_data"

st.set_page_config(page_title="Financial Data Platform", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


st.title("Financial Data Platform")
st.caption("8 actions US + SPY, fenêtre glissante de 10 ans")

prices = load("prices_all")
corr = load("correlation")

st.header("Vue d'ensemble")

tickers = sorted(prices["ticker"].unique())
chosen = st.multiselect("Tickers", tickers, default=["AAPL", "JPM", "SPY"])

# Prix normalisés à 100 au premier jour, pour comparer des actions de niveaux différents
pivot = prices.pivot(index="Date", columns="ticker", values="Close")
if chosen:
    norm = pivot[chosen] / pivot[chosen].iloc[0] * 100
    st.plotly_chart(px.line(norm, labels={"value": "Base 100", "Date": ""}),
                    width="stretch")
else:
    st.info("Choisis au moins un ticker.")

st.subheader("Corrélations des rendements")
st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1), width="stretch")