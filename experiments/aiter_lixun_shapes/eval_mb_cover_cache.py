#!/usr/bin/env python3
"""The cover/split memory-bound rule with a cache exception, CPU only (PYTHONPATH=. python eval_mb_cover_cache.py).

As eval_mb_cover.py (short side covers min(M, N), long side 256 first, split to TARGET busy CUs within K // K_MIN),
except that when B (2 N K bytes, N the long side) is at most CACHE, the short side may drop below the small dimension:
the largest tile (BM from 256 down, then BN from 256 down) that reaches TARGET busy CUs within the K limit. Also: on
REFIT M <= 16, N >= 8192, the best 16-row wide tile (16x128, 16x256) against 16x16x256.
"""
import itertools, math, statistics
import eval_joint_rule as e, eval_joint_rule_mb as m, eval_mb_traffic as tr, eval_mb_cover as c
mb, ts = tr.load_refit_mb(), tr.load_tilesel()
# (a) REFIT M<=16, N>=16384: 16-row wide tiles vs 16x16x256
bt = lambda d,t: min([v for (tt,_),v in d.items() if tt==t] or [math.inf])
rows=[]
for (M,N,K),d in mb.items():
    if M<=16 and N>=8192:
        wide=min(bt(d,t) for t in ('16x256x128','16x128x128','16x128x64'))
        sq=bt(d,'16x16x256'); rows.append((M,N,K,sq/wide))
w=sum(r[3]<0.98 for r in rows); l=sum(r[3]>1.02 for r in rows)
print(f'REFIT M<=16, N>=8192: {len(rows)} shapes; 16x16x256 faster than the best 16x128/16x256 by >2% on {w}, slower on {l}; median time ratio {statistics.median(r[3] for r in rows):.3f}')
for r in sorted(rows,key=lambda r:r[3])[::4]: print('   M=%d N=%d K=%d  16x16x256 / wide = %.2f'%r)
# (b) cover rule with a cache exception: if B (2 N K bytes) <= CACHE, BM may go below the small side
def rule_c(M,N,K,target,k_min,cache):
    if 2*N*K > cache: return c.rule(M,N,K,target,k_min,True)
    for bm in (256,128,64,32,16):                      # largest tile first, BM no longer has to cover M
        if bm > max(16, min(M,256)) * 1: continue
        av=c.tiles_for(bm)
        for bn in c.LONG:
            t=av.get(bn)
            if not t: continue
            T=-(-M//bm)*-(-N//bn); s=1 if T>=target else math.ceil(target/T)
            if s <= max(1,K//k_min): return t,s
    return c.rule(M,N,K,target,k_min,True)
def picker(target,k_min,cache):
    def pick(M,N,K):
        if N<M:
            t,s=rule_c(N,M,K,target,k_min,cache); return c.transpose(t),[m.table_kind_mb(N,M,K,t),'cap']
        t,s=rule_c(M,N,K,target,k_min,cache); return t,[m.table_kind_mb(M,N,K,t),'cap']
    return pick
grid=list(itertools.product((128,192,256),(1024,2048,4096),(0,4<<20,16<<20,64<<20)))
sc=lambda pts,p: m.gm(tr.regret(pts,picker(*p))[0])
fits=sorted((sc(mb,p),p) for p in grid); print('cover + cache exception, top fits (TARGET, K_MIN, CACHE bytes):',[(round(g,3),p) for g,p in fits[:5]])
b=fits[0][1]
loo=[]
for sh in mb:
    rest={s:d for s,d in mb.items() if s!=sh}
    loo+=tr.regret({sh:mb[sh]},picker(*min(grid,key=lambda p: sc(rest,p))))[0]
print('REFIT MB leave one out:',m.stats(loo))
for g,pts in sorted(ts.items()):
    xs,fb=tr.regret(pts,picker(*b)); print(f'tilesel {g}: {len(xs)} shapes {m.stats(xs)}, table kind not timed {fb}')
