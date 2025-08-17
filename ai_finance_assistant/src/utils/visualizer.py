import streamlit as st
import plotly.express as px
import pandas as pd

class PortfolioVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def display_metrics(self, total_return, annualized_volatility, sharpe_ratio):
        st.header("Portfolio Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Return (%)", f"{total_return:.2f}" if total_return is not None else "N/A")
        col2.metric("Annualized Volatility (%)", f"{annualized_volatility:.2f}" if annualized_volatility is not None else "N/A")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if sharpe_ratio is not None else "N/A")
    
    def plot_performance(self, portfolio_returns):
        if portfolio_returns is not None:
            st.header("Portfolio Performance")
            cumulative_returns = (1 + portfolio_returns).cumprod() - 1
            df_plot = pd.DataFrame({"Date": cumulative_returns.index, "Cumulative Return": cumulative_returns})
            fig = px.line(df_plot, x="Date", y="Cumulative Return", title="Portfolio Cumulative Returns")
            fig.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)
    
    def display_composition(self, composition):
        if composition is not None:
            st.header("Portfolio Composition")
            st.dataframe(composition, use_container_width=True)
