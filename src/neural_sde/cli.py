"""Reproducible simulation, fit, and diagnostic plots."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import simulate_heston,load_prices
from .research import Config,experiment


def main(argv=None):
    p=argparse.ArgumentParser(description='Neural SDE versus Gaussian and Heston-style volatility baselines')
    p.add_argument('--csv',help='CSV with date,close adjusted market data')
    p.add_argument('--days',type=int,default=1000)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--rolling',type=int,default=15)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',default='results')
    a=p.parse_args(argv)
    prices=load_prices(a.csv) if a.csv else simulate_heston(days=a.days,seed=a.seed)
    metrics,predictions,diagnostics=experiment(prices,Config(rolling=a.rolling,epochs=a.epochs,hidden=a.hidden,seed=a.seed))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(out/'predictions.csv',index=False)
    report={'source':'user adjusted close CSV' if a.csv else 'synthetic Heston price paths',
            'settings':vars(a),'metrics':metrics,'diagnostics':diagnostics,
            'limitations':'One-step approximate conditional return densities, not exact neural or Heston transition densities; same-close price execution and live trading not evaluated.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    labels=list(metrics)
    axes[0].bar(labels,[metrics[m]['test']['gaussian_nll'] for m in labels]);axes[0].set(title='Held-out Gaussian negative log likelihood',ylabel='NLL (lower is better)')
    axes[1].bar(labels,[metrics[m]['test']['var_95_exceedance'] for m in labels]);axes[1].axhline(.05,color='red',ls='--',label='Nominal 5%')
    axes[1].set(title='Held-out 5% VaR exceedance',ylabel='Observed fraction');axes[1].legend()
    for ax in axes:ax.tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)
    print(json.dumps({'test':{k:v['test'] for k,v in metrics.items()}},indent=2))
    return report

if __name__=='__main__': main()