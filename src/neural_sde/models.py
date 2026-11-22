"""Conditional daily return models (Gaussian Euler transition approximations).

These are conditional observation models, not exact Heston likelihoods. NeuralSDE
parameterizes conditional mu(x) and sigma(x)>0 for dlogS = mu(x)dt+sigma(x)dW.
Historical features x are an observed-state proxy, not a latent continuous path.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import minimize

LOG2PI=np.log(2*np.pi)

def gaussian_nll(y,mean,std):
    y=np.asarray(y,float)
    mean=np.asarray(mean,float)
    std=np.asarray(std,float)
    if y.shape!=mean.shape or y.shape!=std.shape or not np.isfinite(y).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('Invalid Gaussian forecast')
    return float(np.mean(.5*(LOG2PI+2*np.log(std)+((y-mean)/std)**2)))

class GBM:
    """Constant daily drift and volatility, fitted solely to training returns."""
    def fit(self,y):
        y=np.asarray(y,float)
        if y.ndim!=1 or len(y)<20 or not np.isfinite(y).all():raise ValueError('Invalid training returns')
        self.mean=float(y.mean())
        self.std=float(max(y.std(ddof=1),1e-6))
        return self
    def forecast(self,X):
        return np.full(len(X),self.mean),np.full(len(X),self.std)

class HestonFilter:
    """Heston-inspired autoregressive variance proxy using past squared returns.

    This is NOT a calibrated latent Heston state-space likelihood. Squared-return
    variance reversion is a tractable benchmark inspired by the CIR variance SDE.
    """
    def fit(self,returns):
        y=np.asarray(returns,float)
        if y.ndim!=1 or len(y)<30 or not np.isfinite(y).all():raise ValueError('Invalid returns')
        self.mu=float(y.mean())
        self.longvar=float(max(np.mean((y-self.mu)**2),1e-8))
        self.alpha=.93
        best=np.inf
        for a in np.linspace(.25,.99,32):
            history=self.longvar
            loss=0.
            for t in range(1,len(y)):
                history=a*history+(1-a)*(y[t-1]-self.mu)**2
                loss+=(np.log(history+1e-10)+(y[t]-self.mu)**2/(history+1e-10))
            if loss<best:best=loss;self.alpha=float(a)
        self.last_var=self.longvar
        for obs in y:
            self.last_var=self.alpha*self.last_var+(1-self.alpha)*(obs-self.mu)**2
        return self
    def forecast_sequential(self,returns,initial=None):
        """Predict each r[t] using history <t, then update with realized r[t]."""
        y=np.asarray(returns,float)
        if y.ndim!=1 or not np.isfinite(y).all():raise ValueError('Invalid returns')
        v=float(self.last_var if initial is None else initial)
        mean=np.empty(len(y));std=np.empty(len(y))
        for i,obs in enumerate(y):
            mean[i]=self.mu
            std[i]=np.sqrt(max(v,1e-10))
            v=self.alpha*v+(1-self.alpha)*(obs-self.mu)**2
        return mean,std,v

class NeuralSDE(nn.Module):
    def __init__(self,features=4,hidden=32,dt=1/252):
        super().__init__()
        if features<1 or hidden<4 or dt<=0:raise ValueError('Invalid neural parameters')
        self.dt=dt
        self.trunk=nn.Sequential(nn.Linear(features,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.head=nn.Linear(hidden,2)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():self.head.bias.copy_(torch.tensor([0.,-4.5]))
    def forward(self,x):
        raw=self.head(self.trunk(x))
        # Predict daily mean and daily positive volatility; represent annual SDE
        # parameters through mu_day=mu_year*dt and sigma_day=sigma_year*sqrt(dt).
        daily_mean=.05*torch.tanh(raw[...,0])
        daily_std=F.softplus(raw[...,1])+.0002
        return daily_mean,daily_std
    def forecast(self,x):
        self.eval()
        with torch.no_grad():
            mean,std=self(torch.as_tensor(x,dtype=torch.float32))
        return mean.numpy().astype(float),std.numpy().astype(float)
    def fit_model(self,data,epochs=40,lr=.002,seed=42,patience=10):
        if epochs<1 or lr<=0 or patience<1:raise ValueError('Invalid training settings')
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        x=torch.as_tensor(data.features,dtype=torch.float32)
        y=torch.as_tensor(data.returns,dtype=torch.float32)
        optim=torch.optim.Adam(self.parameters(),lr=lr)
        best=float('inf');best_state=None;wait=0
        for epoch in range(epochs):
            self.train()
            idx=torch.randperm(len(data.train))
            for chunk in idx.split(128):
                batch=data.train[chunk.numpy()]
                m,s=self(x[batch]);target=y[batch]
                loss=(torch.log(s)+.5*((target-m)/s)**2).mean()
                optim.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(),1.)
                optim.step()
            self.eval()
            with torch.no_grad():
                m,s=self(x[data.validation]);value=float((torch.log(s)+.5*((y[data.validation]-m)/s)**2).mean())
            if value<best-1e-7:
                best=value;best_state={k:v.detach().clone() for k,v in self.state_dict().items()};wait=0
            else:
                wait+=1
                if wait>=patience:break
        self.load_state_dict(best_state)
        self.eval()
        return {'best_validation_nll_omitting_constant':best,'epochs_run':epoch+1,
                'parameters':sum(p.numel() for p in self.parameters())}