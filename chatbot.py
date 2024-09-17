from flask import Flask, request, jsonify, render_template
import yfinance as yf

app = Flask(__name__)

# Financial data retrieval function
def get_company_financials(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    income_statement = stock.financials
    cash_flow = stock.cashflow
    return balance_sheet, income_statement, cash_flow

# Chatbot endpoint for API requests
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_message = data.get("message")
    
    # Company detection logic
    if "Reliance" in user_message:
        ticker = "RELIANCE.NS"
    elif "Larsen & Toubro" in user_message or "LT" in user_message:
        ticker = "LT.NS"
    else:
        return jsonify({"response": "Company not found!"})
    
    # Get the financial data
    balance_sheet, income_statement, cash_flow = get_company_financials(ticker)
    
    # Convert balance sheet to a readable format (example)
    balance_sheet_str = balance_sheet.to_string()
    
    return jsonify({"response": f"Balance Sheet for {ticker}:\n{balance_sheet_str}"})

# Serve the frontend HTML
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
