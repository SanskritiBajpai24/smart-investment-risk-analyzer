import streamlit as st
import plotly.express as px
from engine import FinancialEngine
import numpy as np

st.set_page_config(page_title="Investment Risk AI", layout="wide")

st.title("🛡️ Smart Investment Risk Analyzer")
st.markdown("---")

ticker = st.sidebar.text_input("Enter Company Ticker", "AAPL").upper()

if st.sidebar.button("Run Risk Analysis"):
    engine = FinancialEngine(ticker)
    
    with st.spinner(f'Fetching financial data for {ticker}...'):
        history_df = engine.get_financial_history()
        prediction, current_z = engine.get_ml_prediction()
        
        # --- DATA INTEGRITY CHECK ---
        if history_df.empty or np.isnan(current_z):
            st.error(f"❌ Incomplete Data for {ticker}")
            st.warning("The API could not retrieve all required balance sheet items. This often happens with very new companies or non-US tickers.")
            st.info("Try a major established ticker like MSFT, AAPL, or TSLA to verify the system.")
        else:
            # --- DISPLAY SECTION ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Altman Z-Score", current_z)
            c2.metric("ML Risk Classification", prediction)
            
            # Use status callouts for clear visual feedback
            if current_z > 3.0:
                c3.success("Financial Status: STRONG")
            elif current_z > 1.8:
                c3.warning("Financial Status: WATCHLIST")
            else:
                c3.error("Financial Status: DISTRESS")

            st.subheader("📊 Financial Health Trend")
            fig = px.area(history_df, x='Date', y='Z_Score', title=f"{ticker} Health over Time")
            fig.add_hline(y=1.8, line_dash="dash", line_color="red", annotation_text="Bankruptcy Risk")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("### Raw Historical Indicators")
            st.dataframe(history_df)
            
            st.markdown("---")
            st.subheader("📁 Analysis Export")
            
            # Convert dataframe to CSV
            csv = history_df.to_csv(index=False).encode('utf-8')
            
            col_dl, col_info = st.columns([1, 2])
            
            with col_dl:
                st.download_button(
                    label="Download Risk Report (CSV)",
                    data=csv,
                    file_name=f"{ticker}_risk_report.csv",
                    mime="text/csv",
                )
            
            with col_info:
                st.info(f"Report generated for {ticker}. The Z-Score trend is based on the last 4 years of SEC-filed financials.")

            # Summary Logic for quick reading
            st.write("### 📝 Quick Verdict")
            if current_z < 1.8:
                st.error(f"⚠️ {ticker} shows high financial distress. Ensure you check their upcoming debt maturity dates.")
            elif 1.8 <= current_z <= 3.0:
                st.warning(f"🔍 {ticker} is in a stable but cautious position. Monitor their EBIT margins closely.")
            else:
                st.success(f"✅ {ticker} is financially robust. A high Z-Score suggests a strong balance sheet.")