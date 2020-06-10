#data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#filter warnings for final presentation
import warnings
warnings.filterwarnings("ignore")
import edhec_risk_kit as erk
import yfinance as yf
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

#notebook formatting
from IPython.core.display import display, HTML

def drawdowns2020(data):
    return_series = pd.DataFrame(data.pct_change().dropna()['2020':])
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return drawdowns.min(axis=0)

def returns_heatmap(data, max_drawdowns, title, reit='No', currencies='No'):
    """
    
    """
    if reit=='Yes':
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:], data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', 'Drawdowns']
        
    elif currencies=='Yes':
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:],  data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(42).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '2-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', 'Drawdowns']
     
    else:
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:], data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], data.pct_change(252*3).iloc[-1,:], max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', '3-Year', 'Drawdowns']
        
        
    df_perf = (df.T*100).sort_values(by='1-Day', ascending=False)
    df_perf.index.name = title
    return df_perf.round(2).style.format('{0:,.2f}%')\
                 .background_gradient(cmap='RdYlGn')\
                 .set_properties(**{'font-size': '10pt',})


def data_sov():
    #Soveriegn Fixed Income ETFs
    data_sov = yf.download('SHY IEF TLT IEI EMB EMLC AGZ BWX IGOV TIP')['Adj Close']
    data_sov.dropna(inplace=True)
    data_sov.columns = ['iShares Agency Bond ETF', 'SPDR International Treasury Bond ETF', 'USD Emerging Markets Bond ETF', 'EM Local Currency Bond ETF', '7-10 Year Treasury Bond ETF','3-7 Year Treasury Bond ETF', 'iShares International Treasury Bond ETF','1-3 Year Treasury Bond ETF', 'US TIPS Bond ETF','20+ Year Treasury Bond ETF']
    return data_sov

def data_corp():
    #Corporate Fixed Income ETFs -  IG & HY in Developed & EM
    data_corp = yf.download('AGG BND BNDX LQD HYG SHYG JNK FALN ANGL FPE HYXE HYXU HYEM EMHY')['Adj Close']
    data_corp.dropna(inplace=True)
    data_corp.columns = ['Core US Aggregate Bond', 'ANGL Fallen Angel HY Bond', 'US Total Bond Market', 'Total International Bond', 
                'iShares EM High Yield','FALN Fallen Angels USD Bond', 'Preferred Securities', 'VanEck EM High Yield',
                'HYG US High Yield', 'US High Yield ex-Energy', 'Int High Yield', 'JNK US High Yield', 
                'US Investment Grade', '0-5Y US High Yield']
    return data_corp

def data_reit(ticker='No'):
    #Real Estate Investment Trust (REIT) ETFs
    data_reit = yf.download('VNQ VNQI SRVR INDS HOMZ REZ PPTY IFEU REM MORT SRET RFI FFR GQRE CHIR FFR WPS IFGL KBWY BBRE ROOF NETL SPG SRG SKT STOR')['Adj Close']['2019':]
    data_reit.dropna(inplace=True)
    if ticker == 'Yes':
        data_reit.columns = data_reit.columns
    else:
        data_reit.columns = ['BetaBuilders', 'China RE', 'DM RE', 'Quality RE', 'ResidentialHOMZ', 'Europe RE', 'IGFL', 'Industrial RE',
                     'YieldEQ RE', 'MORT','NetLease RE', 'Divserified RE', 'MortgageREM', 'ResidentialREZ', 'Cohen RE',
                     'Small-Cap RE', 'TangerRetail', 'SimonRetail', 'SuperDividend', 'SeritageRetail', 'DataInfra RE',
                     'StoreRetail', 'VanguardUS', 'VanguardInt', 'DevelopedRE']
        
    return data_reit

def data_cur():
    #Currencies
    data_cur = yf.download('KRWUSD=X  BRLUSD=X  IDRUSD=X  MXNUSD=X  RUBUSD=X  CADUSD=X  JPYUSD=X  EURUSD=X  INRUSD=X  TRYUSD=X  NZDUSD=X  GBPUSD=X  DX-Y.NYB  AUDUSD=X  AUDJPY=X  EURCHF=X')['Adj Close']['2017':]
    data_cur.dropna(inplace=True)
    data_cur.columns = ['Aussie Yen', 'Australian Dollar', 'Brazilian Real', 'Canadian Dollar', 'Dollar Index', 'EUR/CHF',
                    'Euro', 'British Pound', 'Indonesian Rupiah', 'Indian Rupee', 'Japanese Yen', 'South Korean Won',
                    'Mexican Peso', 'New Zealand Dollar', 'Russian Ruble', 'Turkish Lira']
    return data_cur


def data_comd():
    #Soveriegn Fixed Income ETFs
    data_comd = yf.download('COMT GSG DBC USO CL=F HG=F COPX GC=F GLD GDX PA=F PALL PPLT SI=F SIL ICLN TAN W=F ZC=F NG=F')['Adj Close']
    data_comd.dropna(inplace=True)
    data_comd.columns = ['Crude Oil WTI','COMT', 'Copper Miners', 'DB CMTY Fund', 'Gold Futures', 'Gold Miners',
                     'Gold ETF', 'GSCI ETF', 'Copper Futures', 'Clean Energy', 'NatGas Futures',
                     'Palladium Futures', 'Physical Palladium ETF', 'Physical Platinum ETF', 'Silver Futures', 'Silver ETF', 
                     'Solar ETF', 'USO Oil ETF', 'Wheat Futures', 'Corn Futures']
    return data_comd


def heatmap_fixed_income(days=1, Ticker='No', figsize=(12,6)):
    data_sov1 = yf.download('SHY IEF TLT IEI EMB EMLC AGZ BWX IGOV TIP')['Adj Close']
    data_sov1.dropna(inplace=True)
    data_corp1 = yf.download('AGG BND BNDX LQD HYG SHYG JNK FALN ANGL FPE HYXE HYXU HYEM EMHY')['Adj Close']
    data_corp1.dropna(inplace=True)
    sov_rets = pd.DataFrame(data_sov1.pct_change(days).iloc[-1,:])
    corp_rets = pd.DataFrame(data_corp1.pct_change(days).iloc[-1,:])
    rets = (sov_rets.append(corp_rets))
    rets.columns = ['Return']
    if Ticker == 'Yes':
        rets.index = rets.index
    else:
        rets.index = ['Agency', 'Int-Govt', 'EM Govt', 'EM LCL', '7-10Y UST', '3-7Y UST', 'Int-Govt1', '1-3Y UST',
              'US TIPS', '20Y+ UST', 'Agg Bonds', 'ANGL', 'Total Bonds', 'Int-Bonds', 'EM HY', 'FALN', 'Preferred',
              'HY EM', 'HYG', 'HYXE', 'Int-HY', 'JNK', 'US IG', 'SHYG']
    rets = rets.sort_values('Return', ascending=False)
    symbols = (np.asarray(list(rets.index))).reshape(4,6)
    pct_rets = (np.asarray(rets['Return'].values)).reshape(4,6)
    rows = [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4]
    cols = [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6]
    rets['Rows'] = rows
    rets['Cols'] = cols
    result = rets.pivot(index = 'Rows', columns = 'Cols', values = 'Return')
    labels = (np.asarray(["{0} \n {1:.2%}".format(symb,value)
                     for symb, value in zip(symbols.flatten(), pct_rets.flatten())])).reshape(4,6)
    fig, ax = plt.subplots(figsize=(12,6))
    title = 'Fixed Income ETFs Heatmap'
    plt.title(title, fontsize=15)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    sns.heatmap(result, annot=labels, fmt="", cmap = 'RdYlGn', linewidth=0.30, ax=ax, square=True, annot_kws={"size": 12})
    plt.show()

def heatmap_reit(days=1, Ticker='No', figsize=(12,6)):
    reit = yf.download('VNQ VNQI SRVR INDS HOMZ REZ PPTY IFEU REM MORT SRET RFI FFR GQRE CHIR FFR WPS IFGL KBWY BBRE ROOF NETL SPG SRG SKT STOR')['Adj Close']['2019':]
    reit.dropna(inplace=True)
    if Ticker == 'Yes':
        reit.columns = reit.columns
    else:
        reit.columns = ['BetaBuilders', 'China RE', 'DM RE', 'Quality RE', 'ResidentialHOMZ', 'Europe RE', 'IGFL', 'Industrial RE',
                     'YieldEQ RE', 'MORT','NetLease RE', 'Divserified RE', 'MortgageREM', 'ResidentialREZ', 'Cohen RE',
                     'Small-Cap RE', 'TangerRetail', 'SimonRetail', 'SuperDividend', 'SeritageRetail', 'DataInfra RE',
                     'StoreRetail', 'VanguardUS', 'VanguardInt', 'DevelopedRE']
    rets = pd.DataFrame(reit.pct_change(days).iloc[-1,:])
    rets.columns = ['Return']
    rets = rets.sort_values('Return', ascending=False)
    symbols = (np.asarray(list(rets.index))).reshape(5,5)
    pct_rets = (np.asarray(rets['Return'].values)).reshape(5,5)
    rows = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]
    cols = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
    rets['Rows'] = rows
    rets['Cols'] = cols
    result = rets.pivot(index = 'Rows', columns = 'Cols', values = 'Return')
    labels = (np.asarray(["{0} \n {1:.2%}".format(symb,value)
                     for symb, value in zip(symbols.flatten(), pct_rets.flatten())])).reshape(5,5)
    fig, ax = plt.subplots(figsize=figsize)
    title = 'REIT ETFs Heatmap'
    plt.title(title, fontsize=15)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    sns.heatmap(result, annot=labels, fmt="", cmap = 'RdYlGn', linewidth=0.30, ax=ax, annot_kws={"size": 12})
    plt.show()
    

def heatmap_commodities(days=1, Ticker='No', figsize=(12,6)):
    comd = yf.download('COMT GSG DBC USO CL=F HG=F COPX GC=F GLD GDX PA=F PALL PPLT SI=F SIL ICLN TAN W=F ZC=F NG=F')['Adj Close']
    comd.dropna(inplace=True)
    if Ticker == 'Yes':
        comd.columns = comd.columns
    else:
        comd.columns = ['Crude Oil WTI', 'COMT', 'Copper Miners', 'DB CMTY Fund', 'Gold Futures', 'Gold Miners',
                     'Gold ETF', 'GSCI ETF', 'Copper Futures', 'Clean Energy', 'NatGas Futures',
                     'Palladium Futures', 'Physical Palladium ETF', 'Physical Platinum ETF', 'Silver Futures', 'Silver ETF', 
                     'Solar ETF', 'USO Oil ETF', 'Wheat Futures', 'Corn Futures']
    rets = pd.DataFrame(comd.pct_change(days).iloc[-1,:])
    rets.columns = ['Return']
    rets = rets.sort_values('Return', ascending=False)
    symbols = (np.asarray(list(rets.index))).reshape(4,5)
    pct_rets = (np.asarray(rets['Return'].values)).reshape(4,5)
    rows = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
    cols = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
    rets['Rows'] = rows
    rets['Cols'] = cols
    result = rets.pivot(index = 'Rows', columns = 'Cols', values = 'Return')
    labels = (np.asarray(["{0} \n {1:.2%}".format(symb,value)
                     for symb, value in zip(symbols.flatten(), pct_rets.flatten())])).reshape(4,5)
    fig, ax = plt.subplots(figsize=figsize)
    title = 'Commodities ETF/Futures Heatmap'
    plt.title(title, fontsize=15)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    sns.heatmap(result, annot=labels, fmt="", cmap = 'RdYlGn', linewidth=0.30, ax=ax, annot_kws={"size": 12})
    plt.show()
    
def heatmap_fx(days=1, Ticker='No', figsize=(12,6)):
    cur = yf.download('KRWUSD=X  BRLUSD=X  IDRUSD=X  MXNUSD=X  RUBUSD=X  CADUSD=X  JPYUSD=X  EURUSD=X  INRUSD=X  TRYUSD=X  NZDUSD=X  GBPUSD=X  DX-Y.NYB  AUDUSD=X  AUDJPY=X  EURCHF=X')['Adj Close']['2017':]
    cur.dropna(inplace=True)
    if Ticker == 'Yes':
        cur.columns = cur.columns
    else:
        cur.columns = ['Aussie Yen', 'Australian Dollar', 'Brazilian Real', 'Canadian Dollar', 'Dollar Index', 'EUR/CHF',
                    'Euro', 'British Pound', 'Indonesian Rupiah', 'Indian Rupee', 'Japanese Yen', 'South Korean Won',
                    'Mexican Peso', 'New Zealand Dollar', 'Russian Ruble', 'Turkish Lira']
    
    rets = pd.DataFrame(cur.pct_change(days).iloc[-1,:])
    rets.columns = ['Return']
    rets = rets.sort_values('Return', ascending=False)
    symbols = (np.asarray(list(rets.index))).reshape(4,4)
    pct_rets = (np.asarray(rets['Return'].values)).reshape(4,4)
    rows = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
    cols = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
    rets['Rows'] = rows
    rets['Cols'] = cols
    result = rets.pivot(index = 'Rows', columns = 'Cols', values = 'Return')
    labels = (np.asarray(["{0} \n {1:.2%}".format(symb,value)
                     for symb, value in zip(symbols.flatten(), pct_rets.flatten())])).reshape(4,4)
    fig, ax = plt.subplots(figsize=figsize)
    title = 'REIT ETFs Heatmap'
    plt.title(title, fontsize=15)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    sns.heatmap(result, annot=labels, fmt="", cmap = 'RdYlGn', linewidth=0.30, ax=ax, annot_kws={"size": 12})
    plt.show()
    
    
def heatmap(rets, title='Cross Asset ETFs Heatmap', figsize=(15,10), annot_size=12):
    rets.columns = ['Return']
    rets = rets.sort_values('Return', ascending=False)
    symbols = (np.asarray(list(rets.index))).reshape(10,8)
    pct_rets = (np.asarray(rets['Return'].values)).reshape(10,8)
    rows =[]
    for i in range(1,11):
        rows += list(np.repeat(i,8))

    cols = list(list(np.arange(1,9))*10)
    rets['Rows'] = rows
    rets['Cols'] = cols
    
    result = rets.pivot(index = 'Rows', columns = 'Cols', values = 'Return')
    labels = (np.asarray(["{0} \n {1:.2%}".format(symb,value)
                     for symb, value in zip(symbols.flatten(), pct_rets.flatten())])).reshape(10,8)
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title, fontsize=15)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    sns.heatmap(result, annot=labels, fmt="", cmap = 'RdYlGn', linewidth=0.30, ax=ax, annot_kws={"size": annot_size})
    plt.show()
   
    
    
    







