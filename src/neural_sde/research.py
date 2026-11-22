"""Out-of-sample conditional density benchmarking and distributional diagnostics."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm
from .data import prepare
from .models import GBM,HestonFilter,NeuralSDE,gaussian_nll

@dataclass(frozen=True)
class Config:
    rolling: int=15
    epochs: int=40
    hidden: int=32
    seed: int=42
    train_fraction: float=.60
    val_fraction: float=.20
    lr: float=.002


def evaluate(actual,mean,std):
    if len(actual)==0:raise ValueError('empty evaluation')
    z=(np.asarray(actual)-mean)/std
    pit=norm.cdf(z)
    return {
        'gaussian_nll':gaussian_nll(actual,mean,std),
        'rmse_mean':float(np.sqrt(np.mean((actual-mean)**2))),
        'variance_forecast_mse':float(np.mean(((actual-mean)**2-std**2)**2)),
        'var_95_exceedance':float(np.mean(np.asarray(actual)<mean+std*norm.ppf(.05))),
        'pit_ks_stat':float(kstest(pit,'uniform').statistic),
    }


def experiment(prices,cfg=Config()):
    data=prepare(prices,train_fraction=cfg.train_fraction,validation_fraction=cfg.val_fraction,rolling=cfg.rolling)
    gbm=GBM().fit(data.returns[data.train])
    heston=HestonFilter().fit(data.returns[data.train])
    import torch
    torch.manual_seed(cfg.seed)
    nn=NeuralSDE(features=data.features.shape[1],hidden=cfg.hidden,dt=data.dt)
    train_info=nn.fit_model(data,epochs=cfg.epochs,lr=cfg.lr,seed=cfg.seed)
    results=[]
    metrics={}
    # Heston-filter is carried across validation into test without using future data.
    heston_predictions={}
    start_variance=heston.last_var
    for split in ('validation','test'):
        indices=getattr(data,split)
        hm,hs,start_variance=heston.forecast_sequential(data.returns[indices],start_variance)
        heston_predictions[split]=(hm,hs)
    for name in ('GBM','Heston-style filter','Neural SDE'):
        metrics[name]={}
        for split in ('validation','test'):
            idx=getattr(data,split)
            y=data.returns[idx]
            if name=='GBM':mean,std=gbm.forecast(idx)
            elif name=='Heston-style filter':mean,std=heston_predictions[split]
            else:mean,std=nn.forecast(data.features[idx])
            metrics[name][split]=evaluate(y,mean,std)
            for k,t in enumerate(idx):
                results.append({'date':data.dates[t],'split':split,'model':name,'observed_log_return':float(y[k]),
                                'predicted_log_return':float(mean[k]),'predicted_daily_std':float(std[k])})
    # Model-free GBM Monte Carlo terminal benchmark calibrated only to training.
    seed_rng=np.random.default_rng(cfg.seed+100)
    horizon=20
    s0=float(prices['close'].iloc[-1])
    samples=s0*np.exp(horizon*gbm.mean+gbm.std*np.sqrt(horizon)*seed_rng.standard_normal(10000))
    diagnostics={'GBM_train_daily_mean':gbm.mean,'GBM_train_daily_volatility':gbm.std,
                 'Heston_filter_alpha':heston.alpha,'Heston_filter_train_longrun_variance':heston.longvar,
                 'neural_training':train_info,'MC_20_day_GBM_terminal_quantiles':{
                     str(q):float(np.quantile(samples,q)) for q in (.01,.05,.5,.95,.99)},
                 'MC_reference_is_GBM_not_option_price':True,
                 'splits':{s:len(getattr(data,s)) for s in ('train','validation','test')},
                 'model_note':'Heston-style filter is not a calibrated latent Heston SDE; neural SDE uses conditional Euler Gaussian transitions'}
    return metrics,pd.DataFrame(results),diagnostics