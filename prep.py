"""Data prep + discrete-time person-period risk-set construction for the
megalith-emergence hazard analysis. Numerical conventions:
  - Time is calendar years BP: larger = older. Chronological forward time = decreasing BP.
  - A region is 'at risk' of first megalith emergence while BP > emergence date D.
  - Event (emergence) occurs in the time-bin containing D; region then exits the risk set.
  - NA regions never have the event: all their bins are event=0 (right-censored).
"""
import numpy as np, pandas as pd

BIN = 250  # yr

def load_clean(path):
    """Reconstruct the TSV, repairing embedded-newline corruption in `country`."""
    raw = open(path).read().split('\n')
    recs, buf = [], ''
    for line in raw:
        buf = line if buf == '' else buf + '\n' + line
        if buf.count('\t') >= 9:
            recs.append(buf); buf = ''
    header = recs[0].split('\t')
    df = pd.DataFrame([r.split('\t') for r in recs[1:]], columns=header)
    for c in ['p','se','res_norm','ageAverage','megalith_emerge']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['country'] = df['country'].str.replace('\n','/',regex=False)
    return df

def wide_set(df, comb):
    """Return sample-level wide table for one set: one row/sample, ancestry cols = sources."""
    d = df[df.comb_idx==comb]
    wide = d.pivot_table(index='sample_id', columns='source_pop', values='p')
    meta = d.drop_duplicates('sample_id').set_index('sample_id')[['country','ageAverage','megalith_emerge']]
    sources = list(wide.columns)
    w = meta.join(wide)
    return w, sources

def build_person_periods(w, sources, region_dates=None, bin_w=BIN):
    """Construct region x bin person-period rows.
    region_dates: optional dict country->emergence date (for permutation nulls);
                  default uses the observed megalith_emerge.
    Returns DataFrame with columns: country, bin_mid, event, imputed, n_samples, baseline time, + sources.
    """
    w = w.copy()
    w['bin'] = (np.floor(w['ageAverage']/bin_w)*bin_w + bin_w/2)  # bin midpoint (BP)
    if region_dates is None:
        region_dates = w.dropna(subset=['megalith_emerge']).groupby('country')['megalith_emerge'].first().to_dict()
    rows = []
    for country, g in w.groupby('country'):
        D = region_dates.get(country, np.nan)  # NaN => never-event (censored)
        binstats = g.groupby('bin').agg(n=('ageAverage','size'))
        anc = g.groupby('bin')[sources].mean()
        obs_bins = sorted(anc.index.tolist(), reverse=True)  # old -> young (decreasing BP)
        if len(obs_bins)==0:
            continue
        def nearest_anc(target_bin):
            i = int(np.argmin([abs(b-target_bin) for b in obs_bins]))
            return anc.loc[obs_bins[i]].values, obs_bins[i]
        has_event = not np.isnan(D)
        event_bin = (np.floor(D/bin_w)*bin_w + bin_w/2) if has_event else None
        for b in obs_bins:
            if has_event and b < event_bin:
                continue  # observed bin after emergence -> not part of the risk process
            event = int(has_event and b == event_bin)
            rows.append({'country':country,'bin_mid':b,'event':event,'imputed':0,
                         'n_samples':int(binstats.loc[b,'n']),
                         **{s:anc.loc[b,s] for s in sources}})
        if has_event and (event_bin not in obs_bins):
            vals, src_bin = nearest_anc(event_bin)
            rows.append({'country':country,'bin_mid':event_bin,'event':1,'imputed':1,
                         'n_samples':0, **{s:vals[j] for j,s in enumerate(sources)}})
    pp = pd.DataFrame(rows)
    pp['time_c'] = (pp['bin_mid'] - pp['bin_mid'].mean())/1000.0  # centered kyr for baseline
    return pp

if __name__=='__main__':
    import sys
    df = load_clean(sys.argv[1])
    w, src = wide_set(df, 'set_1')
    pp = build_person_periods(w, src)
    print('set_1 person-periods:', pp.shape, '| events:', int(pp.event.sum()), '| imputed rows:', int(pp.imputed.sum()))
    print(pp.groupby('country').agg(periods=('event','size'), events=('event','sum'), imputed=('imputed','sum')).to_string())
