"""Firth-penalized discrete-time hazard model + region-level permutation test.
Response: per-region-bin megalith-emergence event (hazard). Covariate: region-bin ancestry.
Variant A: baseline = intercept only (marginal). Variant B: baseline = intercept + linear calendar time (time-controlled).
Screening statistic: per-source Rao score statistic (stable at 6 events, scale-invariant).
Effect size: Firth-penalized coefficient per SD. p-values by permuting region emergence-date labels."""
import numpy as np, pandas as pd
from prep import load_clean, wide_set, BIN

def firth_logit(X, y, max_iter=1000, tol=1e-8):
    n,p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X@beta; eta=np.clip(eta,-30,30)
        pi = 1/(1+np.exp(-eta)); W = pi*(1-pi)
        XtWX = X.T@(X*W[:,None])
        try: inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError: inv = np.linalg.pinv(XtWX)
        h = W*np.einsum('ij,ij->i', X@inv, X)
        U = X.T@(y - pi + h*(0.5-pi))
        step = inv@U
        pll0 = _pll(X,y,beta)
        t=1.0
        for _ in range(40):
            nb=beta+t*step
            if _pll(X,y,nb) >= pll0 - 1e-12: break
            t*=0.5
        beta = beta + t*step
        if np.max(np.abs(t*step))<tol: break
    return beta, _pll(X,y,beta)

def _pll(X,y,beta):
    eta=np.clip(X@beta,-30,30); pi=1/(1+np.exp(-eta))
    ll=np.sum(y*np.log(pi+1e-300)+(1-y)*np.log(1-pi+1e-300))
    s,ld=np.linalg.slogdet(X.T@(X*(pi*(1-pi))[:,None]))
    return ll+0.5*ld

def build_cache(w, sources, bin_w=BIN):
    """Per-region: observed bins (old->young), mean ancestry, n. Independent of date assignment."""
    w=w.copy(); w['bin']=np.floor(w['ageAverage']/bin_w)*bin_w+bin_w/2
    cache={}
    for c,g in w.groupby('country'):
        anc=g.groupby('bin')[sources].mean()
        n=g.groupby('bin').size()
        order=np.argsort(-anc.index.values)  # decreasing BP
        cache[c]={'bins':anc.index.values[order],'anc':anc.values[order],'n':n.values[order]}
    obs_dates=w.dropna(subset=['megalith_emerge']).groupby('country')['megalith_emerge'].first().to_dict()
    return cache, obs_dates

def assemble(cache, date_dict, bin_w=BIN):
    """Return design pieces: A (n x nsrc ancestry), tvec (calendar time centered kyr), y (event)."""
    A=[]; tv=[]; y=[]
    for c,cc in cache.items():
        D=date_dict.get(c,np.nan); has=not (D is None or np.isnan(D))
        ebin=(np.floor(D/bin_w)*bin_w+bin_w/2) if has else None
        bins=cc['bins']; anc=cc['anc']
        seen_ebin=False
        for i,b in enumerate(bins):
            if has and b<ebin: continue
            ev=int(has and b==ebin)
            if ev: seen_ebin=True
            A.append(anc[i]); tv.append(b); y.append(ev)
        if has and not seen_ebin:  # impute event bin from nearest observed
            j=int(np.argmin(np.abs(bins-ebin)))
            A.append(anc[j]); tv.append(ebin); y.append(1)
    A=np.array(A); tv=np.array(tv); y=np.array(y,float)
    tv=(tv-tv.mean())/1000.0
    return A, tv, y

def _logit_fit(Z, y, max_iter=200, tol=1e-10):
    """Plain (unpenalized) logistic Newton fit for the baseline nuisance model (stable: no separation)."""
    beta=np.zeros(Z.shape[1])
    for _ in range(max_iter):
        eta=np.clip(Z@beta,-30,30); pi=1/(1+np.exp(-eta)); W=pi*(1-pi)
        g=Z.T@(y-pi); H=Z.T@(Z*W[:,None])
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: step=np.linalg.pinv(H)@g
        beta=beta+step
        if np.max(np.abs(step))<tol: break
    eta=np.clip(Z@beta,-30,30); pi=1/(1+np.exp(-eta))
    return beta, pi

def screen(A, tv, y, variant='A'):
    """Per-source Rao score statistic for adding one source to the baseline nuisance model.
    Evaluated at the baseline fit -> no separation/divergence, scale-invariant in x.
    Returns (signed score chi-sq [nsrc], sign [nsrc]); rank by magnitude."""
    n=len(y); ones=np.ones((n,1))
    Z = ones if variant=='A' else np.column_stack([ones,tv])
    _, pi0 = _logit_fit(Z, y)
    W0 = pi0*(1-pi0); r = y - pi0
    ZtW = Z.T*W0                      # k x n
    C = ZtW@Z                          # k x k
    Cinv = np.linalg.pinv(C)
    nsrc=A.shape[1]; stats=np.empty(nsrc); signs=np.empty(nsrc)
    for j in range(nsrc):
        x=A[:,j]
        U = x@r
        Bv = ZtW@x
        V = (x*W0)@x - Bv@Cinv@Bv
        if V<=1e-12:
            stats[j]=0.0; signs[j]=0.0
        else:
            stats[j]=U*U/V; signs[j]=np.sign(U)
    return stats*signs, np.sign(signs)

def firth_effect(A, tv, y, j, variant='A'):
    """Robust Firth coefficient for source j (standardized covariate) for effect-direction reporting."""
    n=len(y); ones=np.ones((n,1))
    base = ones if variant=='A' else np.column_stack([ones,tv])
    x=A[:,j]; sd=x.std()
    xs=(x-x.mean())/sd if sd>0 else x*0.0
    b,_=firth_logit(np.column_stack([base,xs]),y)
    return b[-1]  # per-1-SD change in ancestry, log-odds of emergence hazard

def run_set(df, comb, variant='A', n_perm=1000, seed=0):
    w,src=wide_set(df,comb); cache,obs=build_cache(w,src)
    A,tv,y=assemble(cache,obs)
    stats_s,_=screen(A,tv,y,variant)          # signed score statistic
    absstat=np.abs(stats_s)
    betas=np.array([firth_effect(A,tv,y,j,variant) for j in range(len(src))])
    # permutation: shuffle the multiset of emergence dates (incl NaN) across regions (region-level exchangeability)
    countries=list(cache.keys())
    date_vals=np.array([obs.get(c,np.nan) for c in countries])
    rng=np.random.default_rng(seed)
    perm_max=np.empty(n_perm); perm_abs=np.empty((n_perm,len(src)))
    for p in range(n_perm):
        dd={c:v for c,v in zip(countries,rng.permutation(date_vals))}
        Ap,tvp,yp=assemble(cache,dd)
        sp,_=screen(Ap,tvp,yp,variant); a=np.abs(sp)
        perm_abs[p]=a; perm_max[p]=np.nanmax(a)
    p_marg=np.array([(1+np.sum(perm_abs[:,j]>=absstat[j]))/(1+n_perm) for j in range(len(src))])
    p_fw=np.array([(1+np.sum(perm_max>=absstat[j]))/(1+n_perm) for j in range(len(src))])
    from scipy.stats import chi2
    res=pd.DataFrame({'source':src,'firth_beta_perSD':betas,'score_chi2':absstat,
                      'p_chi2_asym':chi2.sf(absstat,1),'p_perm':p_marg,'p_perm_FWER':p_fw})
    res=res.sort_values('score_chi2',ascending=False).reset_index(drop=True)
    return res, {'n_pp':len(y),'n_events':int(y.sum()),'obs_max':float(absstat.max())}

if __name__=='__main__':
    import sys, os
    path=sys.argv[1]
    nperm=int(sys.argv[2]) if len(sys.argv)>2 else 5000
    outdir=sys.argv[3] if len(sys.argv)>3 else '.'
    df=load_clean(path)
    for v,label in [('A','MARGINAL (no time control)'),('B','TIME-CONTROLLED')]:
        allr=[]
        print('\n'+'='*70+f'\nVARIANT {v}: {label}\n'+'='*70)
        for s in ['set_1','set_2','set_3','set_4','set_5','set_6']:
            res,info=run_set(df,s,variant=v,n_perm=nperm,seed=42)
            res.insert(0,'set',s); allr.append(res)
            print(f"\n{s}  (person-periods={info['n_pp']}, events={info['n_events']})")
            print(res.head(3).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
        out=pd.concat(allr)
        fn=os.path.join(outdir,f'results_{v}.csv')
        out.to_csv(fn,index=False)
        print('\nsaved',fn)
