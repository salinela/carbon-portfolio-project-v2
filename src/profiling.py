import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def profile_numeric(df, cols, by=None):
    """Per-group: missingness, sign, robust percentiles, skew, orders-of-magnitude span."""
    def _stats(s):
        s = pd.to_numeric(s, errors='coerce'); v = s.dropna(); pos = v[v > 0]
        return pd.Series({
            'n': len(s), 'miss%': round(100*s.isna().mean(),1),
            'zero': int((v==0).sum()), 'neg': int((v<0).sum()),
            'min': v.min(), 'p1': v.quantile(.01), 'p50': v.median(),
            'p99': v.quantile(.99), 'max': v.max(), 'skew': round(v.skew(),1),
            'oom': round(np.log10(pos.quantile(.99)/pos.quantile(.01)),1) if len(pos)>1 else np.nan,
        })
    rows=[]
    for c in cols:
        if by is None:
            r=_stats(df[c]); r.name=c; rows.append(r)
        else:
            for g,sub in df.groupby(by):
                r=_stats(sub[c]); r.name=f"{c} [{g}]"; rows.append(r)
    return pd.DataFrame(rows)

def plot_dist(df, col, by='source', bins=50):
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    for g,sub in df.groupby(by):
        x = pd.to_numeric(sub[col], errors='coerce'); x = x[x>0]
        ax[0].hist(np.log10(x), bins=bins, alpha=.5, label=str(g))
    ax[0].set(xlabel=f'log10({col})', ylabel='count', title=f'{col} by {by}'); ax[0].legend()
    groups=[(str(g), np.log10(pd.to_numeric(sub[col],errors='coerce').pipe(lambda s:s[s>0])))
            for g,sub in df.groupby(by)]
    ax[1].boxplot([v for _,v in groups], tick_labels=[g for g,_ in groups])
    ax[1].set(ylabel=f'log10({col})', title=f'{col} spread by {by}')
    plt.tight_layout(); plt.show()

