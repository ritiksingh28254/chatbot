import yfinance as yf

def get_company_financials(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    income_statement = stock.financials
    cash_flow = stock.cashflow
    return balance_sheet, income_statement, cash_flow

import yfinance as yf

def get_company_financials(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    income_statement = stock.financials
    cash_flow = stock.cashflow
    return balance_sheet, income_statement, cash_flow

# Extract financials for Reliance Industries
balance_sheet, income_statement, cash_flow = get_company_financials("LT.NS")

# Display the extracted financial data
print("Balance Sheet:\n", balance_sheet)
print("\nIncome Statement:\n", income_statement)
print("\nCash Flow Statement:\n", cash_flow)




# def calculate_roe(net_income, shareholders_equity):
#     return (net_income / shareholders_equity) * 100

# def calculate_roce(ebit, capital_employed):
#     return (ebit / capital_employed) * 100

# def get_company_indicators(ticker):
#     # Retrieve financial data using the existing function
#     balance_sheet, income_statement, _ = get_company_financials(ticker)
    
#     # Access the latest data (assuming the most recent period is at the first position)
#     net_income = income_statement.loc['Net Income'].values[0]
#     total_shareholder_equity = balance_sheet.loc['Total Stockholder Equity'].values[0]
    
#     # Calculate Capital Employed
#     total_assets = balance_sheet.loc['Total Assets'].values[0]
#     total_current_liabilities = balance_sheet.loc['Total Current Liabilities'].values[0]
#     capital_employed = total_assets - total_current_liabilities
    
#     # EBIT might be listed as 'Earnings Before Interest and Taxes'
#     ebit = income_statement.loc['Ebit'].values[0]
    
#     # Calculate financial ratios
#     roe = calculate_roe(net_income, total_shareholder_equity)
#     roce = calculate_roce(ebit, capital_employed)
    
#     # Return calculated values
#     return {'ROE': roe, 'ROCE': roce}

# # Example usage
# indicators = get_company_indicators("LT.NS")
# print("ROE:", indicators['ROE'])
# print("ROCE:", indicators['ROCE'])