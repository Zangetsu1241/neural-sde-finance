import json
import numpy as np
import pandas as pd
import pytest
from neural_sde.data import simulate_heston,load_prices,prepare
from neural_sde.models import GBM,HestonFilter,NeuralSDE,gaussian_nll
from neural_sde.research import Config,evaluate,experiment
from neural_sde.cli import main


def test_simulation_is_reproducible_and_positive():
    a=simulate_heston(days=150,seed=12)
    b=simulate_heston(days=150,seed=12)
    pd.testing.assert_frame_equal(a,b)
    assert (a.close>0).all() and (a.latent_variance>=0).all()


def test_heston_degenerate_noise_is_deterministic():
    a=simulate_heston(140,seed=1,xi=0,rho=0)
    b=simulate_heston(140,seed=1,xi=0,rho=0)
    assert np.array_equal(a.latent_variance,b.latent_variance)


def test_features_are_causal():
    original=simulate_heston(280)
    modified=original.copy()
    modified.loc[modified.index[230:],'close']*=2
    a=prepare(original)
    b=prepare(modified)
    np.testing.assert_allclose(a.features[a.train],b.features[b.train])
    np.testing.assert_allclose(a.returns[a.train],b.returns[b.train])
    np.testing.assert_allclose(a.mean,b.mean)


def test_split_is_chronological():
    d=prepare(simulate_heston(250))
    assert d.train[-1]<d.validation[0] and d.validation[-1]<d.test[0]
    assert len(d.returns)==len(d.features)==len(d.dates)


def test_log_return_target_alignment():
    p=simulate_heston(220)
    d=prepare(p,rolling=15)
    for i in [0,len(d.returns)-1]:
        j=p.index.get_loc(d.dates[i])
        assert np.isclose(d.returns[i],np.log(p.close.iloc[j]/p.close.iloc[j-1]))


def test_gbm_constant_forecast():
    model=GBM().fit(np.linspace(-.02,.02,100))
    m,s=model.forecast(np.arange(10))
    assert np.allclose(m,0) and np.all(s>0) and np.allclose(s,s[0])


def test_heston_filter_updates_only_after_predicting():
    model=HestonFilter().fit(np.random.default_rng(9).normal(0,.01,200))
    future=np.array([.01,.02,-.04,.01])
    m,a,_=model.forecast_sequential(future)
    modified=future.copy();modified[-1]=.8
    m2,b,_=model.forecast_sequential(modified)
    np.testing.assert_allclose(a,b)
    assert np.all(a>0)


def test_neural_density_positive():
    m=NeuralSDE(hidden=8)
    mu,s=m.forecast(np.zeros((5,4),dtype=float))
    assert mu.shape==(5,) and np.all(s>0)


def test_neural_training_produces_finite_scores():
    data=prepare(simulate_heston(240))
    m=NeuralSDE(hidden=8)
    report=m.fit_model(data,epochs=2,seed=15)
    mean,std=m.forecast(data.features[data.test])
    assert np.isfinite(gaussian_nll(data.returns[data.test],mean,std))
    assert report['parameters']>0


def test_training_is_reproducible():
    d=prepare(simulate_heston(230,seed=8))
    import torch
    torch.manual_seed(7)
    a=NeuralSDE(hidden=8);ra=a.fit_model(d,epochs=2,seed=9)
    torch.manual_seed(7)
    b=NeuralSDE(hidden=8);rb=b.fit_model(d,epochs=2,seed=9)
    np.testing.assert_allclose(a.forecast(d.features[d.test])[0],b.forecast(d.features[d.test])[0])
    assert ra==rb


def test_density_rejects_invalid_scales():
    with pytest.raises(ValueError):gaussian_nll(np.zeros(3),np.zeros(3),np.zeros(3))


def test_var_exceedance_is_calculated_correctly():
    x=np.array([-5.,0.,0.,0.])
    e=evaluate(x,np.zeros(4),np.ones(4))
    assert e['var_95_exceedance']==.25


def test_bad_csv_is_rejected(tmp_path):
    path=tmp_path/'bad.csv'
    path.write_text('date,close\n2020-01-01,100\n2020-01-01,110\n')
    with pytest.raises(ValueError):load_prices(path)


def test_full_benchmark_completes():
    metrics,pred,details=experiment(simulate_heston(240),Config(epochs=2,hidden=8))
    assert set(metrics)=={'GBM','Heston-style filter','Neural SDE'}
    assert len(pred)==(details['splits']['validation']+details['splits']['test'])*3
    assert np.isfinite(pred.predicted_daily_std).all()
    assert set(pred.split)=={'validation','test'}


def test_cli_outputs(tmp_path):
    main(['--days','220','--epochs','2','--hidden','8','--output',str(tmp_path)])
    for f in ('predictions.csv','metrics.json','diagnostics.png'):
        assert (tmp_path/f).exists()
    assert len(json.loads((tmp_path/'metrics.json').read_text())['metrics'])==3