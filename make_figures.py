"""Generate supplement figures for the megalith-emergence hazard analysis.
  Figure 1 (fig_strength.png): per-set best-source association strength (permutation FWER).
  Figure 2 (fig_panels.png): 12 panels = for each of the 6 source sets, (b) the set's most
     strongly associated source averaged by region, and (c) that source's trajectory over time.
  Panel (b): megalith regions use post-emergence samples only; non-megalith regions use all samples.
  Panel (c): blue = megalith regions after their own emergence date; grey = non-megalith regions.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from prep import load_clean

TSV='megalith_mixtureModelling_emegergenceDates.tsv'
BLUE='#2980b9'; GREY='#bdc3c7'; RED='#c0392b'

# per-set most strongly associated source (from results_B.csv) and its ancestry label
TOP={'set_1':('0_4_1_3_2_2800+','WHG (Western Hunter-Gatherer)'),
     'set_2':('0_3_3_4_1_1_2_2800+','Cardial Impressed Ware'),
     'set_3':('0_3_1_2_2_2800+','Western Cardial-related'),
     'set_4':('0_3_1_1_1_2_1_2800+','North French-related'),
     'set_5':('0_3_1_1_1_2_1_2800+','North French-related'),
     'set_6':('0_3_1_1_1_2_1_2800+','North French-related')}
SHORT={'Portugal/Spain':'Iberia','France_South':'France_S','France_North':'France_N',
       'Italy_Sardinia':'Sardinia','BritishIsles/Ireland':'Britain','Denmark/Sweden':'Denmark',
       'Italy':'Italy','Latvia/Lithuania/Estonia':'Baltic',
       'Bulgaria/NorthMacedonia/Romania/Albania/Serbia':'Balkans','Hungary':'Hungary',
       'Greece/Croatia':'Greece','Austria/Poland/CzechRepublic':'CentralEur'}

df=load_clean(TSV)

# ---------------- Figure 1: association strength per set ----------------
B=pd.read_csv('results_B.csv')
best=B.sort_values('score_chi2',ascending=False).groupby('set').first().reset_index().sort_values('set')
fig,ax=plt.subplots(figsize=(7,4.5))
cols=[RED if p<0.05 else GREY for p in best.p_perm_FWER]
ax.bar(best.set,-np.log10(best.p_perm_FWER),color=cols)
ax.axhline(-np.log10(0.05),ls='--',c='k',lw=1)
ax.text(-0.4,-np.log10(0.05)+0.03,'$P_{FWER}=0.05$',fontsize=9)
for i,row in enumerate(best.itertuples()):
    ax.text(i,-np.log10(row.p_perm_FWER)+0.03,row.source.split('_2800')[0],
            rotation=90,ha='center',va='bottom',fontsize=7)
ax.set_ylabel('$-\\log_{10} P_{FWER}$ (best source)')
ax.set_title('Association strength of the best source per set (time-controlled hazard)')
plt.tight_layout(); plt.savefig('fig_strength.png',dpi=150); plt.close()
print('saved fig_strength.png')

# ---------------- Figure 2: 12 panels (b,c per set) ----------------
fig,axes=plt.subplots(6,2,figsize=(12,20))
emerge_dates=[5600,5900,6300,6700]
for r,s in enumerate(['set_1','set_2','set_3','set_4','set_5','set_6']):
    src,label=TOP[s]
    d=df[(df.comb_idx==s)&(df.source_pop==src)].copy()
    d['meg']=d['megalith_emerge'].notna()
    d['post']=d['meg'] & (d['ageAverage']<d['megalith_emerge'])  # younger than emergence = megaliths present

    # ---- panel b: region means (megalith regions -> post-emergence samples only) ----
    axb=axes[r,0]; means={}; colr={}
    for c,g in d.groupby('country'):
        meg=g['meg'].iloc[0]
        use=g[g['post']] if meg else g
        if len(use)==0: continue
        means[SHORT[c]]=use['p'].mean(); colr[SHORT[c]]=BLUE if meg else GREY
    ser=pd.Series(means).sort_values()
    axb.barh(range(len(ser)),ser.values,color=[colr[k] for k in ser.index])
    axb.set_yticks(range(len(ser))); axb.set_yticklabels(ser.index,fontsize=8)
    axb.set_xlabel('mean proportion',fontsize=8)
    axb.set_title(f'({s}) b — {src.split("_2800")[0]}\n{label}',fontsize=9)

    # ---- panel c: trajectory (blue = megalith regions post-emergence; grey = non-megalith) ----
    axc=axes[r,1]
    blue=d[d['post']].copy(); grey=d[~d['meg']].copy()
    for sub,col,lab in [(blue,BLUE,'with megaliths'),(grey,GREY,'without megaliths')]:
        if len(sub)==0: continue
        sub['bin']=np.floor(sub['ageAverage']/250)*250+125
        mt=sub.groupby('bin')['p'].mean()
        axc.plot(mt.index,mt.values,'o-',c=col,ms=4,label=lab)
    for dd in emerge_dates: axc.axvline(dd,ls=':',c='k',lw=0.5,alpha=0.5)
    axc.invert_xaxis(); axc.set_xlabel('years BP',fontsize=8)
    axc.set_ylabel('mean proportion',fontsize=8)
    axc.set_title(f'({s}) c — {label} over time',fontsize=9)
    if r==0: axc.legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig_panels.png',dpi=150); plt.close()
print('saved fig_panels.png')
