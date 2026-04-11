"""
Анализ 2026 года: старые параметры (bars=3, breach=3%) vs новые (bars=1, breach=1.5%)
Показывает по месяцам, какие сделки были/не были бы, сколько заработал.
"""
import sys, io, json, os, requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_START='2018-01-01'; TEST_START='2026-01-01'; TEST_END='2026-04-11'
START_BALANCE=10000.0  # реалистичный счёт к 2026
RISK_PCT=0.01; COMMISSION=2.0; MAX_POSITIONS=4
SMA_FAST,SMA_SLOW=50,200; ATR_LEN,RS_LEN,VOL_LEN=14,63,20
MAX_ATR_PCT=5.0; MIN_PRICE=20; MIN_VOL=500_000
EXCLUDE_SECTORS={'Real Estate','Utilities'}; TOP_SECTORS=5
EARNINGS_BUFFER=7; HOURLY_DISC=0.01
B_RECOVERY_BODY=0.5; B_RECOVERY_VOL=1.2; B_STOP_BUFFER=0.5
SR_MIN=20; SR_MAX=80

SECTOR_ETF={
    'Information Technology':'XLK','Financials':'XLF','Energy':'XLE',
    'Health Care':'XLV','Industrials':'XLI','Consumer Discretionary':'XLY',
    'Consumer Staples':'XLP','Materials':'XLB','Communication Services':'XLC'
}
EARNINGS_CACHE='d:/projects/trading/earnings_cache.json'

CONFIGS = [
    ('Старый (bars=3, breach=3.0%)', 3, 3.0),
    ('Новый  (bars=1, breach=1.5%)', 1, 1.5),
]

def calc_atr(high,low,close,period=14):
    tr=pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
    return tr.ewm(span=period,adjust=False).mean()

def load_earnings():
    if os.path.exists(EARNINGS_CACHE):
        raw=json.load(open(EARNINGS_CACHE,encoding='utf-8'))
        return {s:set(v) for s,v in raw.items()}
    return {}

def build_sector_leaders(spx_close):
    etfs=list(SECTOR_ETF.values())
    raw=yf.download(etfs,start=DATA_START,end='2027-01-01',progress=False,auto_adjust=True,group_by='ticker')
    sec_prices={}
    for sector,etf in SECTOR_ETF.items():
        try: sec_prices[sector]=raw[etf]['Close'].squeeze()
        except: pass
    idx=spx_close.index; leaders={}
    for i in range(RS_LEN,len(idx)):
        date=idx[i]; spx_ret=spx_close.iloc[i]/spx_close.iloc[i-RS_LEN]; scores={}
        for sec,prices in sec_prices.items():
            p=prices.reindex(idx,method='ffill'); p_now,p_ago=p.iloc[i],p.iloc[i-RS_LEN]
            if pd.isna(p_now) or pd.isna(p_ago) or p_ago==0: continue
            scores[sec]=(p_now/p_ago)/spx_ret
        if len(scores)>=TOP_SECTORS:
            leaders[date]=set(sorted(scores,key=scores.get,reverse=True)[:TOP_SECTORS])
    return leaders

def signals_type_b(close,high,low,open_,volume,atr,vol_ma,sma50,sma200,spx,rs,b_bars,b_breach):
    prior_high=high.shift(SR_MIN).rolling(SR_MAX-SR_MIN).max()
    local_min=low.rolling(15).min().shift(1)
    recent_low=low.shift(1).rolling(b_bars).min()
    false_break_ph=(recent_low<prior_high)&(recent_low>=prior_high*(1-b_breach/100))
    false_break_lm=(recent_low<local_min)&(recent_low>=local_min*(1-b_breach/100))
    false_break=false_break_ph|false_break_lm
    recovery_ph=(close>prior_high)&(low.shift(1)<prior_high)
    recovery_lm=(close>local_min)&(low.shift(1)<local_min)
    strong_recovery=(close>open_)&((close-open_)/atr>B_RECOVERY_BODY)&(volume>vol_ma*B_RECOVERY_VOL)
    recovered=(
        ((recovery_ph|recovery_lm)&false_break)|
        ((close>prior_high*0.999)&(close.shift(1)<prior_high)&false_break_ph)|
        ((close>local_min*0.999)&(close.shift(1)<local_min)&false_break_lm)
    )
    trend_ok=(close>sma200)&(sma50>sma200)
    quality_ok=((atr/close*100)<MAX_ATR_PCT)&(rs>1.0)&(vol_ma>MIN_VOL)&(close>MIN_PRICE)
    market_ok=spx>spx.rolling(200).mean()
    return trend_ok&quality_ok&false_break&recovered&strong_recovery&market_ok

def run_symbol(sym,df_stock,spx_close,earnings,sec_leaders,stock_sec,b_bars,b_breach):
    try:
        if isinstance(df_stock.columns,pd.MultiIndex):
            df_stock.columns=df_stock.columns.get_level_values(0)
        close=df_stock['Close'].squeeze(); high=df_stock['High'].squeeze()
        low=df_stock['Low'].squeeze(); open_=df_stock['Open'].squeeze()
        volume=df_stock['Volume'].squeeze()
        if len(close)<220: return []
        sma50=close.rolling(SMA_FAST).mean(); sma200=close.rolling(SMA_SLOW).mean()
        atr=calc_atr(high,low,close,ATR_LEN); vol_ma=volume.rolling(VOL_LEN).mean()
        spx=spx_close.reindex(close.index,method='ffill')
        rs=(close/close.shift(RS_LEN))/(spx/spx.shift(RS_LEN))
        earn_set=earnings.get(sym,set())
        mask=(close.index>=TEST_START)&(close.index<=TEST_END)
        sig_b=signals_type_b(close,high,low,open_,volume,atr,vol_ma,sma50,sma200,spx,rs,b_bars,b_breach)
        bars=list(close.index); results=[]
        for sig_date in sig_b[mask&sig_b].index:
            sig_d=sig_date.date()
            if any(0<=(pd.to_datetime(ed).date()-sig_d).days<=EARNINGS_BUFFER for ed in earn_set): continue
            if stock_sec and sig_date in sec_leaders:
                if stock_sec not in sec_leaders[sig_date]: continue
            idx_i=bars.index(sig_date)
            entry_price=close.iloc[idx_i]*(1-HOURLY_DISC)
            lookback=min(b_bars+1,idx_i)
            recent_lows=low.iloc[idx_i-lookback:idx_i]
            breakdown_low=recent_lows.min() if len(recent_lows)>0 else low.iloc[idx_i]
            sl_init=breakdown_low-atr.iloc[idx_i]*B_STOP_BUFFER
            risk=entry_price-sl_init; risk_pct=risk/entry_price*100
            if risk<=0 or risk_pct>15: continue
            res_cands=[high.iloc[max(0,idx_i-lb):idx_i].max() for lb in [10,20,40,60]
                       if high.iloc[max(0,idx_i-lb):idx_i].max()>entry_price*1.01]
            tp=entry_price+risk*(max(1.5,min(5.0,(min(res_cands)-entry_price)/risk)) if res_cands else 3.0)
            actual_rr=(tp-entry_price)/risk
            if actual_rr<1.0: continue
            exit_p,reason,hold=None,'timeout',0
            for j in range(1,21):
                if idx_i+j>=len(bars): break
                bh=high.iloc[idx_i+j]; bl=low.iloc[idx_i+j]; bc=close.iloc[idx_i+j]
                bs=sma50.iloc[idx_i+j]; ba=atr.iloc[idx_i+j]; hold=j
                if bl<=sl_init: exit_p=sl_init; reason='stop'; break
                if bh>=tp:      exit_p=tp;      reason='target'; break
                if bc<bs-ba*0.5:exit_p=bc;      reason='sma50'; break
            if exit_p is None:
                exit_p=close.iloc[min(idx_i+hold,len(close)-1)]
            pnl_pct=(exit_p-entry_price)/entry_price*100
            results.append({
                'symbol':sym,'date':sig_date,'sector':stock_sec or '',
                'entry':round(entry_price,2),'exit':round(exit_p,2),
                'sl':round(sl_init,2),'tp':round(tp,2),
                'rr':round(actual_rr,2),'risk_pct':round(risk_pct,2),
                'pnl_pct':round(pnl_pct,2),'win':exit_p>=entry_price,
                'reason':reason,'days':hold,
                'month':sig_date.strftime('%Y-%m'),
            })
        return results
    except: return []

def simulate_account(trades_df, start_bal):
    df=trades_df.sort_values('date').copy()
    account,open_pos,rows=start_bal,[],[]
    for _,t in df.iterrows():
        e_date=pd.to_datetime(t['date']); x_date=e_date+pd.Timedelta(days=int(t['days'])+1)
        open_pos=[d for d in open_pos if d>e_date]
        if len(open_pos)>=MAX_POSITIONS: continue
        open_pos.append(x_date)
        pos=(account*RISK_PCT)/(t['risk_pct']/100)
        pnl_usd=pos*(t['pnl_pct']/100)-COMMISSION
        account+=pnl_usd
        rows.append({**t.to_dict(),'pnl_$':round(pnl_usd,2),'account':round(account,2)})
    return pd.DataFrame(rows), account

def main():
    print("=== Анализ 2026: старые vs новые параметры ===\n")

    print("Загружаем данные...")
    resp=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                      headers={'User-Agent':'Mozilla/5.0'},timeout=15)
    df_wiki=pd.read_html(io.StringIO(resp.text))[0]
    df_wiki=df_wiki[~df_wiki['GICS Sector'].isin(EXCLUDE_SECTORS)]
    symbols=df_wiki['Symbol'].str.replace('.', '-', regex=False).tolist()
    sector_map=dict(zip(df_wiki['Symbol'].str.replace('.', '-', regex=False),df_wiki['GICS Sector']))

    spx_raw=yf.download('SPY',start=DATA_START,end='2027-01-01',progress=False,auto_adjust=True)
    if isinstance(spx_raw.columns,pd.MultiIndex): spx_raw.columns=spx_raw.columns.get_level_values(0)
    spx_close=spx_raw['Close'].squeeze()
    sec_leaders=build_sector_leaders(spx_close)
    earnings=load_earnings()

    all_data={}
    for i in range(0,len(symbols),50):
        batch=symbols[i:i+50]
        raw=yf.download(batch,start=DATA_START,end='2027-01-01',progress=False,auto_adjust=True,group_by='ticker')
        for sym in batch:
            try: all_data[sym]=raw[sym].dropna() if len(batch)>1 else raw.dropna()
            except: pass
        print(f"  [{min(i+50,len(symbols))}/{len(symbols)}]",end=' ',flush=True)
    print()

    all_results = {}
    for label,b_bars,b_breach in CONFIGS:
        trades=[]
        for sym,df in all_data.items():
            t=run_symbol(sym,df.copy(),spx_close,earnings,sec_leaders,sector_map.get(sym),b_bars,b_breach)
            trades.extend(t)
        df_t=pd.DataFrame(trades)
        if len(df_t)==0: continue
        df_t['date']=pd.to_datetime(df_t['date'])
        sim,final=simulate_account(df_t,START_BALANCE)
        all_results[label]=(df_t,sim,final)

    # ── Сравнение итогов ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  2026 (янв–апр): ИТОГ")
    print(f"{'='*70}")
    print(f"  Старт счёта: ${START_BALANCE:,.0f}")
    print(f"  {'':35} {'Сд':>4} {'WIN':>4} {'LOSS':>5} {'Win%':>6} {'P&L $':>8} {'P&L %':>7}")
    print(f"  {'-'*65}")
    for label,(df_t,sim,final) in all_results.items():
        n=len(sim); wins=sim['win'].sum(); losses=n-wins
        wr=sim['win'].mean()*100; pnl=final-START_BALANCE; pnl_pct=pnl/START_BALANCE*100
        print(f"  {label:<35} {n:>4}  {wins:>3}  {losses:>4}  {wr:>5.1f}%  ${pnl:>+6,.0f}  {pnl_pct:>+6.1f}%")

    # ── По месяцам ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ПО МЕСЯЦАМ")
    print(f"{'='*70}")
    months = ['2026-01','2026-02','2026-03','2026-04']
    month_names = {'2026-01':'Январь','2026-02':'Февраль','2026-03':'Март','2026-04':'Апрель'}

    for m in months:
        print(f"\n  ── {month_names[m]} ──")
        for label,(df_t,sim,final) in all_results.items():
            mg=sim[sim['date'].dt.strftime('%Y-%m')==m] if len(sim)>0 else pd.DataFrame()
            if len(mg)==0:
                print(f"    {label}: нет сделок")
            else:
                wr=mg['win'].mean()*100; pnl=mg['pnl_$'].sum()
                print(f"    {label}: {len(mg)}сд  WR{wr:.0f}%  P&L ${pnl:+,.0f}")

    # ── Сделки новых параметров: полный список ────────────────────
    new_label = list(all_results.keys())[1]
    df_new, sim_new, _ = all_results[new_label]

    print(f"\n{'='*70}")
    print(f"  НОВЫЕ ПАРАМЕТРЫ — все сделки 2026")
    print(f"{'='*70}")
    print(f"  {'Дата':<11} {'Тик':<6} {'Сектор':<25} {'Вход':>7} {'Выход':>7} {'R:R':>4} {'P&L%':>6} {'Итог':<8} {'Дни':>4}")
    print(f"  {'-'*85}")

    account=START_BALANCE
    for _,t in sim_new.sort_values('date').iterrows():
        win_str = 'WIN ✓' if t['win'] else 'loss ✗'
        sector_short = t.get('sector','')[:22] if t.get('sector') else ''
        print(f"  {str(t['date'])[:10]}  {t['symbol']:<6} {sector_short:<25} "
              f"${t['entry']:>6.2f} ${t['exit']:>6.2f} {t['rr']:>4.1f} "
              f"{t['pnl_pct']:>+5.1f}%  {win_str:<8} {t['days']:>3}д")

    # ── Сделки которые были в старом но нет в новом ──────────────
    old_label = list(all_results.keys())[0]
    df_old, sim_old, _ = all_results[old_label]

    old_keys = set(zip(sim_old['symbol'], sim_old['date'].dt.date))
    new_keys = set(zip(sim_new['symbol'], sim_new['date'].dt.date))

    removed = old_keys - new_keys
    added   = new_keys - old_keys

    print(f"\n{'='*70}")
    print(f"  ОТФИЛЬТРОВАНО (были в старом, нет в новом): {len(removed)}")
    print(f"{'='*70}")
    for sym,dt in sorted(removed):
        row = sim_old[(sim_old['symbol']==sym) & (sim_old['date'].dt.date==dt)]
        if len(row)>0:
            r=row.iloc[0]
            res='WIN' if r['win'] else 'LOSS'
            print(f"  {str(dt)[:10]}  {sym:<6}  {r['pnl_pct']:>+5.1f}%  {res}  ({r['reason']}, {r['days']}д)")

    print(f"\n{'='*70}")
    print(f"  ДОБАВЛЕНО (есть в новом, не было в старом): {len(added)}")
    print(f"{'='*70}")
    for sym,dt in sorted(added):
        row = sim_new[(sim_new['symbol']==sym) & (sim_new['date'].dt.date==dt)]
        if len(row)>0:
            r=row.iloc[0]
            res='WIN' if r['win'] else 'LOSS'
            print(f"  {str(dt)[:10]}  {sym:<6}  {r['pnl_pct']:>+5.1f}%  {res}  ({r['reason']}, {r['days']}д)")

    print("\nГотово.")

if __name__=='__main__':
    main()
