import yfinance as yf
import pandas as pd
import numpy as np
import joblib

class FinancialEngine:
    def __init__(self, ticker_symbol):
        self.symbol = ticker_symbol.upper()
        self.ticker = yf.Ticker(self.symbol)
        try:
            self.model = joblib.load('risk_model.pkl')
        except:
            self.model = None

    def get_financial_history(self):
        """Fetch historical balance sheet and financials with robust error handling."""
        bs = self.ticker.balance_sheet
        fin = self.ticker.financials
        
        if bs.empty or fin.empty:
            return pd.DataFrame()
        
        # Merge data and fill missing values with 0 to avoid 'nan'
        df = pd.concat([bs, fin]).T.fillna(0)
        
        history = []
        for index, row in df.iterrows():
            try:
                # Use a standard list of potential names for financial items
                ta = row.get('Total Assets', row.get('Assets', 0))
                if ta == 0: continue
                
                ca = row.get('Current Assets', 0)
                cl = row.get('Current Liabilities', 0)
                ebit = row.get('EBIT', 0)
                rev = row.get('Total Revenue', row.get('Revenue', 0))
                re = row.get('Retained Earnings', 0)
                
                # Altman Z-Score Formula for Public Manufacturers
                x1 = (ca - cl) / ta
                x2 = re / ta
                x3 = ebit / ta
                x4 = 1.1  # Simplified coefficient for stability
                x5 = rev / ta
                
                z = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)
                
                if not np.isnan(z):
                    history.append({
                        "Date": index.date() if hasattr(index, 'date') else index, 
                        "Z_Score": round(z, 2)
                    })
            except Exception as e:
                print(f"Row error: {e}")
                continue
                
        return pd.DataFrame(history)

    def get_ml_prediction(self):
        """Returns the ML risk category and current Z-Score."""
        history_df = self.get_financial_history()
        info = self.ticker.info
        
        if history_df.empty:
            return "No Data", 0.0

        latest_z = history_df.iloc[0]['Z_Score']
        
        # Safe extraction for ML features
        curr_ratio = info.get('currentRatio', 1.0)
        # Convert percentages to decimals if needed
        debt_equity = info.get('debtToEquity', 0.0)
        debt_equity = debt_equity / 100 if debt_equity > 10 else debt_equity
        rev_growth = info.get('revenueGrowth', 0.0)

        if self.model:
            features = np.array([[latest_z, curr_ratio, debt_equity, rev_growth]])
            prediction = self.model.predict(features)[0]
            return prediction, latest_z
            
        return "Model Not Loaded", latest_z