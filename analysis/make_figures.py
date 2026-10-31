"""Draws the result figures from results/ (run collect2.py, agg_ext.py and ext15_tests.py first); writes PNG files to analysis/figs/."""
import json,glob,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,PowerNorm
from matplotlib.lines import Line2D
from pathlib import Path
HERE=Path(__file__).resolve().parent; R=HERE.parent/'results'; OUT=HERE/'figs'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':['Times New Roman','DejaVu Serif'],'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
 'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold','mathtext.fallback':'stix','font.size':11,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8a8f98','grid.color':'#e6e7ea','legend.frameon':False,'savefig.dpi':300})
N=json.load(open(HERE/'numbers2.json')); A=json.load(open(HERE/'agg_ext.json')); E15=json.load(open(R/'ext15_tests.json'))
DSN={'beauty':'Amazon-Beauty','lastfm':'Last-FM','ml10m':'MovieLens-10M'}; DS=list(DSN)
INK='#1d2733'; B1,B2,B3='#2a78d6','#eb6834','#1baf7a'; C={'x':'#9aa5b1','n':'#5b8fd6','f':'#d1495b'}
m_=lambda d,v,m: N[f'{d}.{v}.{m}'][0]
BL=['UserCF','ItemCF','NCF','DeepFM','LightGCN','SGL','SimGCL','LightGCL','xsim','HCCF','SEPT']; LAB=lambda v:'XSimGCL' if v=='xsim' else v
# relative improvement heatmap
mat=np.array([[(m_(d,'full',m)/m_(d,v,m)-1)*100 for d in DS for m in 'PRN'] for v in BL])
fig,ax=plt.subplots(figsize=(8.6,5.6)); cmap=LinearSegmentedColormap.from_list('s',['#eef4fc','#9cc3ee','#2a78d6','#123f7a'])
im=ax.imshow(mat,cmap=cmap,norm=PowerNorm(.45,vmin=0,vmax=mat.max()),aspect='auto')
for i in range(mat.shape[0]):
    for j in range(9): ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=8.3,color='white' if mat[i,j]>30 else INK)
ax.set_xticks(range(9)); ax.set_xticklabels(['P@10','R@10','N@10']*3); ax.set_yticks(range(len(BL))); ax.set_yticklabels([LAB(v) for v in BL])
for x in (2.5,5.5): ax.axvline(x,color='white',lw=3)
for j,d in enumerate(DS): ax.text(j*3+1,-.95,DSN[d],ha='center',va='bottom',fontweight='bold')
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]; ax.set_ylim(len(BL)-.5,-1.3)
cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.02); cb.set_label('Relative improvement of SGC-Rec (%)'); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(OUT/'improvement_heatmap.png'); plt.close()
# accuracy-diversity
fig,axs=plt.subplots(1,3,figsize=(12,3.7))
for k,(ax,d) in enumerate(zip(axs,DS)):
    for v in ['LightGCN','SGL','SimGCL','LightGCL','xsim','HCCF','SEPT','full']:
        hi=v=='full'; ax.scatter(N[f'{d}.{v}.E'],m_(d,v,'R'),s=260 if hi else 60,c=B2 if hi else (B1 if v=='xsim' else '#9aa1ab'),marker='*' if hi else 'o',edgecolor='white',zorder=4)
        ax.annotate('SGC-Rec' if hi else LAB(v),(N[f'{d}.{v}.E'],m_(d,v,'R')),xytext=(6,4),textcoords='offset points',fontsize=8.5,fontweight='bold' if hi else 'normal')
    ax.set_title(f"({'abc'[k]}) {DSN[d]}"); ax.set_xlabel('Shannon entropy of top-10 categories (nats)'); ax.grid(True); ax.set_axisbelow(True)
axs[0].set_ylabel('Recall@10'); fig.tight_layout(); fig.savefig(OUT/'accuracy_diversity.png'); plt.close()
# user-activity groups (Amazon-Beauty)
sp=N['sparsity']; fig,ax=plt.subplots(1,3,figsize=(14,4))
for k,(a,(m,t)) in enumerate(zip(ax,[('P','Precision@10'),('R','Recall@10'),('N','NDCG@10')])):
    x=np.arange(4); xv=[sp[str(g)]['x'+m] for g in range(4)]; fv=[sp[str(g)]['f'+m] for g in range(4)]
    bs=[a.bar(x-.215,xv,.4,color=C['x'],label='XSimGCL'),a.bar(x+.215,fv,.4,color=C['f'],label='SGC-Rec')]; top=max(xv+fv)
    for bars in bs:
        for r in bars: a.text(r.get_x()+r.get_width()/2,r.get_height()+top*.012,f'{r.get_height():.4f}',ha='center',va='bottom',fontsize=7)
    a.set_ylim(0,top*1.12); a.set_xticks(x); a.set_xticklabels(['<10','10–20','20–50','>50']); a.set_xlabel('training interactions'); a.set_title(f"({'abc'[k]}) {t}")
ax[0].legend(loc='upper left'); plt.tight_layout(); plt.savefig(OUT/'user_activity_groups.png'); plt.close()
# training scarcity
def catgain(d,k):
    D=[json.load(open(f)) for f in sorted(glob.glob(str(R/f'scx_{d}_s*_k{k}.json')))]
    return 100*(np.mean([x['cat_prior_only']['R'] for x in D])/np.mean([x['cat_nopre_only']['R'] for x in D])-1)
panels={'Catalogue expert:\npre-trained vs. not':lambda d,k:catgain(d,k),'Fused model:\npre-trained vs. not':lambda d,k:100*(A[f'scarce_{d}_{k}']['full_R']/A[f'scarce_{d}_{k}']['nopre_R']-1),'SGC-Rec vs. XSimGCL':lambda d,k:A[f'scarce_{d}_{k}']['full_R_gain%']}
fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharey=True); x=np.arange(3)
for k,(ax,(t,fn)) in enumerate(zip(axs,panels.items())):
    for off,kp,c in ((-.19,20,B1),(.19,40,B3)):
        vals=[fn(d,kp) for d in DS]; bars=ax.bar(x+off,vals,.36,color=c,label=f'{kp}% of training interactions',zorder=3)
        for r in bars: v=r.get_height(); ax.text(r.get_x()+r.get_width()/2,v+(.8 if v>=0 else -.8),f'{v:+.1f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8.2)
    ax.axhline(0,color='#8a8f98',lw=.8); ax.set_xticks(x); ax.set_xticklabels(['Amazon-\nBeauty','Last-FM','MovieLens-\n10M']); ax.set_title(f"({'abc'[k]}) {t}",fontsize=10.5); ax.grid(axis='y'); ax.set_axisbelow(True)
axs[0].set_ylabel('Relative change of Recall@10 (%)'); axs[0].legend(fontsize=8.5); fig.tight_layout(); fig.savefig(OUT/'training_scarcity.png'); plt.close()
# robustness to injected edges (clean point = seeds 0-2 with the clean-data head)
fig,ax=plt.subplots(1,3,figsize=(12.5,3.7)); xs=[0,10,20,30]
for k,(a,d) in enumerate(zip(ax,DS)):
    M3=[json.load(open(R/f'main_{d}_s{s}.json')) for s in range(3)]
    for v,l,c,mk in [('xsim','XSimGCL',C['x'],'o'),('nopre','SGC-Rec w/o Gaussian pre-training',C['n'],'s'),('full','SGC-Rec',C['f'],'D')]:
        a.plot(xs,[np.mean([m[v]['R'] for m in M3])]+[A[f'noise_{d}_{n}'][v+'_R'] for n in (10,20,30)],marker=mk,color=c,lw=2,label=l)
    a.set_title(f"({'abc'[k]}) {DSN[d]}"); a.set_xlabel('injected training edges (%)'); a.set_xticks(xs); a.grid(True)
ax[0].set_ylabel('Recall@10'); ax[0].legend(fontsize=9,loc='lower left'); plt.tight_layout(); plt.savefig(OUT/'noise_robustness.png'); plt.close()
# ablation
vs=[('xsim','interaction expert'),('cat_nopre_only','catalogue expert\n(no pre-training)'),('cat_prior_only','catalogue expert\n(pre-trained)'),('nopre','w/o pre-training'),('initonly','init.-only prior'),('full_kappa0','$\\kappa=0$'),('novarpath','no variance path'),('full','SGC-Rec')]
cols=[C['x'],'#c9d6e8','#8fb3e0',C['n'],'#e9b44c','#8fbf8f','#b58fd6',C['f']]; fig,ax=plt.subplots(1,3,figsize=(14,4.4))
for k,(a,d) in enumerate(zip(ax,DS)):
    ys=[N[f'{d}.{v}.R'][0] for v,_ in vs]; e=[N[f'{d}.{v}.R'][1]/np.sqrt(5) for v,_ in vs]
    a.bar(range(len(vs)),ys,color=cols,yerr=e,capsize=3,zorder=3); a.set_ylim(min(ys)-3*max(e)-.004,max(ys)+.006)
    a.set_xticks(range(len(vs))); a.set_xticklabels([l for _,l in vs],rotation=45,ha='right',fontsize=9); a.set_title(f"({'abc'[k]}) {DSN[d]}"); a.grid(axis='y'); a.set_axisbelow(True)
ax[0].set_ylabel('Recall@10 (five-seed mean ± s.e.)'); plt.tight_layout(); plt.savefig(OUT/'ablation.png'); plt.close()
# fifteen-seed pre-training effect
E={(r[0],r[1],r[2]):r for r in E15}; fig,axs=plt.subplots(1,2,figsize=(10.5,3.4),sharey=True)
for k,(ax,(m,t)) in enumerate(zip(axs,[('R','Recall@10'),('N','NDCG@10')])):
    for off,cmp,c in ((-.15,'expert',B1),(.15,'fused',B2)):
        for di,d in enumerate(DS):
            r=E[d,m,cmp]; v=r[3]; sig=r[8]<.05; y=di+off
            ax.plot([0,v],[y,y],color=c,lw=2,alpha=.55); ax.scatter(v,y,s=80,facecolor=c if sig else 'white',edgecolor=c,linewidth=2,zorder=4)
            ax.text(v+(.09 if v>=0 else -.09),y,f'{int(r[4])}/15',va='center',ha='left' if v>=0 else 'right',fontsize=8.2)
    ax.axvline(0,color='#8a8f98',lw=.9); ax.set_yticks(range(3)); ax.set_yticklabels([DSN[d] for d in DS]); ax.set_xlim(-.8,3.2)
    ax.set_xlabel(f'Relative change of {t} from pre-training (%)'); ax.set_title(f"({'ab'[k]}) {t}"); ax.grid(axis='x')
axs[0].invert_yaxis()
fig.legend(handles=[Line2D([],[],marker='o',color=B1,lw=2,label='catalogue expert'),Line2D([],[],marker='o',color=B2,lw=2,label='fused SGC-Rec'),Line2D([],[],marker='o',color='#52514e',lw=0,label='Holm-adjusted p < 0.05'),Line2D([],[],marker='o',color='#52514e',markerfacecolor='white',lw=0,label='Holm-adjusted p ≥ 0.05')],loc='lower center',ncol=4,fontsize=8.8)
fig.tight_layout(rect=(0,.08,1,1)); fig.savefig(OUT/'pretraining_effect_15seeds.png'); plt.close()
# sensitivity to kappa and catalogue horizon (validation)
for fn,(xkey,xl) in (('kappa_sensitivity.png',('kappa','$\\kappa$')),('catalogue_horizon.png',('horizon','catalogue-training epochs'))):
    fig,ax=plt.subplots(1,3,figsize=(12.5,3.6))
    for k,(a,d) in enumerate(zip(ax,DS)):
        S=N['sens'][d]
        if xkey=='kappa':
            ks=S['kappas']; xs_=range(len(ks)); f=np.mean([r['full'] for r in S['rows']],0); n=np.mean([r['nopre'] for r in S['rows']],0); sel=ks.index(N['cfg'][d]['kappa'])
            a.set_xticks(list(xs_)); a.set_xticklabels([str(v) for v in ks],fontsize=9)
        else:
            xs_=[r['epochs'] for r in S['horizon']]; f=[r['full'] for r in S['horizon']]; n=[r['nopre'] for r in S['horizon']]; sel=4; a.set_xticks(xs_)
        a.plot(xs_,f,'o-',color=C['f'],lw=2,label='SGC-Rec'); a.plot(xs_,n,'s--',color=C['n'],lw=2,label='w/o Gaussian pre-training'); a.axvline(sel,ls=':',color='k')
        a.set_xlabel(xl); a.set_title(f"({'abc'[k]}) {DSN[d]}"); a.grid(True)
    ax[0].set_ylabel('validation Recall@10'); ax[0].legend(fontsize=9); plt.tight_layout(); plt.savefig(OUT/fn); plt.close()
# computational cost (five-seed means of the measured stage times)
T={d:N[f'{d}.time'] for d in DS}; fig,axs=plt.subplots(1,2,figsize=(12,4.3),gridspec_kw=dict(width_ratios=[1.65,1])); ax=axs[0]; y=np.arange(3); left=np.zeros(3); hs=[]
for key,c,lab in (('xsim','#8fa3bf','XSimGCL interaction expert'),('prior',B3,'Gaussian pre-training'),('cat',B1,'Catalogue expert'),('head',B2,'Gaussian decision head')):
    vals=np.array([T[d][key] for d in DS]); hs.append(ax.barh(y,vals,left=left,color=c,edgecolor='white',height=.56,label=lab,zorder=3))
    for yi,(l,v) in enumerate(zip(left,vals)): ax.text(l+v/2,yi,f'{v:.0f}s',ha='center',va='center',fontsize=8.5,color=INK if key=='xsim' else 'white')
    left+=vals
for yi,d in enumerate(DS): ax.text(left[yi]+1.5,yi,f"{T[d]['total_sgc']:.0f} s ({T[d]['total_sgc']/T[d]['xsim']:.2f}×)",va='center',fontsize=9,fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels([DSN[d] for d in DS]); ax.invert_yaxis(); ax.set_xlim(0,125); ax.set_xlabel('End-to-end training time (s, five-seed mean)'); ax.set_title('(a) Training-time composition'); ax.grid(axis='x')
ax=axs[1]; x=np.arange(3)
for off,key,c,lab in ((-.2,'peak_mb_xsim','#8fa3bf','XSimGCL'),(.2,'peak_mb_sgc',B2,'SGC-Rec')):
    bars=ax.bar(x+off,[T[d][key] for d in DS],.36,color=c,label=lab,zorder=3)
    for r in bars: ax.text(r.get_x()+r.get_width()/2,r.get_height()+8,f'{r.get_height():.0f}',ha='center',va='bottom',fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([DSN[d] for d in DS],fontsize=9); ax.set_ylabel('Peak GPU memory (MB)'); ax.set_title('(b) Peak GPU memory'); ax.set_ylim(0,500); ax.legend(loc='upper left'); ax.grid(axis='y')
fig.legend(handles=hs,labels=[h.get_label() for h in hs],loc='lower center',ncol=4,fontsize=9,bbox_to_anchor=(.36,0)); fig.tight_layout(rect=(0,.09,1,1)); fig.savefig(OUT/'training_cost.png'); plt.close()
print('figures written to',OUT)
