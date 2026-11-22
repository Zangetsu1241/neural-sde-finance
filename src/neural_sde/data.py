"""Synthetic Heston dynamics, point-in-time price ingestion and chronological features."""
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class MarketData:
    returns: np.ndarray
    features: np.ndarray
    dates: pd.Index
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    dt: float

def simulate_heston(days=1000, seed=42, s0=100., mu=.06, kappa=4., theta=.035, xi=.45, rho=-.65, v0=.035, steps_per_day=4):
    """Full-truncation Euler Heston in continuous years; log price integration.

    Brownian increments correlated at each substep. Output closes and latent variance
    for *diagnostics only*, never provided to predictive models.
    """
    if days<120 or steps_per_day<1 or not isinstance(steps_per_day,int) or s0<=0 or theta<=0 or v0<0 or kappa<=0 or xi<0 or not -1<=rho<=1:
        raise ValueError('Invalid simulation parameters')
    rng=np.random.default_rng(seed)
    dt=1/(252*steps_per_day)
    logprice=np.log(s0)
    v=float(v0)
    prices=np.empty(days)
    variance=np.empty(days)
    for day in range(days):
        for _ in range(steps_per_day):
            z1,z2=rng.standard_normal(2)
            z2=rho*z1+np.sqrt(max(0.,1-rho*rho))*z2
            vp=max(v,0.)
            logprice+=(mu-.5*vp)*dt+np.sqrt(vp*dt)*z1
            v=max(v+kappa*(theta-vp)*dt+xi*np.sqrt(vp*dt)*z2,0.)
        prices[day]=np.exp(logprice)
        variance[day]=v
    dates=pd.bdate_range('2018-01-01',periods=days)
    return pd.DataFrame({'close':prices,'latent_variance':variance},index=dates)

def load_prices(path):
    frame=pd.read_csv(path)
    if not {'date','close'}.issubset(frame.columns):
        raise ValueError('CSV requires date,close')
    dates=pd.to_datetime(frame['date'],utc=True,errors='raise')
    close=pd.to_numeric(frame['close'],errors='raise').to_numpy(float)
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing or len(close)<120 or not np.isfinite(close).all() or (close<=0).any():
        raise ValueError('Need >=120 ordered, unique, positive finite closes')
    return pd.DataFrame({'close':close},index=dates)

def prepare(prices, train_fraction=.6, validation_fraction=.2, rolling=15):
    if not .4<=train_fraction<=.8 or not .05<=validation_fraction<=.3 or train_fraction+validation_fraction>.85 or rolling<5:
        raise ValueError('Invalid split or rolling size')
    c=prices['close'].to_numpy(float)
    if len(c)<120 or not np.isfinite(c).all() or np.any(c<=0) or not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError('Invalid prices')
    r=np.diff(np.log(c))
    # Information x[t] constructed from r[:t], target r[t].
    raw=[]
    for t in range(rolling,len(r)):
        history=r[t-rolling:t]
        raw.append([history[-1],np.sqrt(np.mean(history**2)),history.mean(),
                    np.sqrt(np.mean(history[-5:]**2))])
    X=np.asarray(raw,float)
    y=r[rolling:].copy()
    dates=prices.index[rolling+1:]
    n=len(y);a=int(n*train_fraction);b=int(n*(train_fraction+validation_fraction))
    if min(a,b-a,n-b)<20:raise ValueError('Insufficient samples in split')
    mean=X[:a].mean(axis=0)
    scale=X[:a].std(axis=0)
    scale=np.where(scale<1e-10,1.,scale)
    return MarketData(y,(X-mean)/scale,dates,np.arange(a),np.arange(a,b),np.arange(b,n),mean,scale,1/252)