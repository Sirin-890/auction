import numpy as np, itertools, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng=np.random.default_rng(0)
N=100000
grid=np.round(np.arange(0,1.0001,0.01),2)
strats={"truthful":lambda v:v,"half":lambda v:v/2,"random":lambda v:rng.random(v.shape)}
def util(v,b,rb):  # rb: array of rival max bid
    return np.where(b>rb, v-rb, 0.0).mean()
out=[]
# Part 1 A and C
for nr in (1,2):
    fig,ax=plt.subplots(3,3,figsize=(13,10),sharey='row')
    for i,v in enumerate((0.2,0.5,0.8)):
        for j,(sn,f) in enumerate(strats.items()):
            us=[]
            for b in grid:
                vals=rng.random((N,nr)); bids=f(vals)
                us.append(util(v,b,bids.max(axis=1)))
            us=np.array(us); ax[i,j].plot(grid,us); ax[i,j].axvline(v,color='r',ls='--')
            ax[i,j].set_title(f"v={v}, rival: {sn}"); ax[i,j].set_xlabel("bid");ax[i,j].set_ylabel("avg utility")
            out.append(f"{nr} rival(s) v={v} {sn}: argmax bid={grid[us.argmax()]:.2f} (max util {us.max():.4f}), util at b=v {us[int(round(v*100))]:.4f}")
    fig.suptitle(f"Part 1{'A' if nr==1 else 'C'}: {nr} rival(s); red = truthful bid"); fig.tight_layout(); fig.savefig(f"part1_{nr}rival.png",dpi=110)
# Part 1 B
out.append("\nPart 1B (fixed rival bid; ties at b=rival give 0 utility in expectation-equivalent)")
for nr in (1,2):
  for v in (0.2,0.5,0.8):
    for r in (0.1,0.4,0.9):
        u=np.array([max(v-r,0) if b>r else 0 for b in grid])
        mx=u.max(); best=grid[u>=mx-1e-12]
        out.append(f"{nr} rival(s) v={v} rival bid={r}: max util {mx:.2f}, optimal bids [{best.min():.2f},{best.max():.2f}], b=v optimal: {u[int(round(v*100))]>=mx-1e-12}")
# Part 2
out.append("\nPart 2 first-price")
ks=np.round(np.arange(0.1,2.0001,0.1),1)
fig,ax=plt.subplots(1,3,figsize=(15,4))
for a,n in zip(ax,(2,3,4)):
    us=[];fr=[]
    for k in ks:
        v=rng.random((N,n)); bids=v*(n-1)/n; bids[:,0]=k*v[:,0]
        w=bids.argmax(1); srt=np.sort(bids,1); top,sec=srt[:,-1],srt[:,-2]
        us.append(np.where(w==0,v[:,0]-top,0).mean())
        fr.append((top-sec>1e-12).mean())  # winner could have bid just above 2nd bid
    us=np.array(us)
    a.plot(ks,us);a.axvline((n-1)/n,color='r',ls='--');a.set_title(f"n={n}: E[utility] of bidder 1");a.set_xlabel("k")
    out.append(f"n={n}: best k={ks[us.argmax()]} (eq k={(n-1)/n:.3f}); fraction winner could pay less: {np.mean(fr):.4f}")
    # equilibrium case, all bid equilibrium, mean overpay
    v=rng.random((N,n)); b=v*(n-1)/n; s=np.sort(b,1)
    out.append(f"   all-equilibrium: fraction could pay less={(s[:,-1]>s[:,-2]).mean():.4f}, mean overpayment={(s[:,-1]-s[:,-2]).mean():.4f} (theory 1/(n+1)*(n-1)/n={(n-1)/n/(n+1):.4f})")
    for k,u_,f_ in zip(ks,us,fr): out.append(f"   k={k:.1f} util={u_:.4f} frac_less={f_:.4f}")
fig.tight_layout();fig.savefig("part2.png",dpi=110)
# Part 3
E={"AB":("A","B",3),"AC":("A","C",2),"BD":("B","D",2),"BE":("B","E",1),"CF":("C","F",5),"CE":("C","E",3),"DF":("D","F",2),"EF":("E","F",1)}
def best(cost,excl=None):
    paths=[["AB","BD","DF"],["AB","BE","EF"],["AC","CF"],["AC","CE","EF"]]
    c=[(sum(cost[e] for e in p),p) for p in paths if excl is None or excl not in p]
    return min(c)
true={e:x[2] for e,x in E.items()}
def vcg(rep):
    c,p=best(rep); pay={}
    for e in rep:
        if e in p: pay[e]=best(rep,e)[0]-(c-rep[e])
        else: pay[e]=0
    return p,c,pay
p,c,pay=vcg(true); out.append(f"\nPart 3: path {''.join(p[0][0]+''.join(x[1] for x in p))} cost {c} payments {pay}")
fig,ax=plt.subplots(1,2,figsize=(11,4))
for a,e in zip(ax,("BE","CE")):
    rs=np.arange(0,10.01,0.5);us=[]
    for r in rs:
        rep=dict(true);rep[e]=r;p,_,pay=vcg(rep)
        us.append(pay[e]-true[e] if e in p else 0)
    a.step(rs,us,where='post');a.axvline(true[e],color='r',ls='--');a.set_title(f"edge {e} (true {true[e]}), "+("on path" if e=="BE" else "off path"));a.set_xlabel("reported cost");a.set_ylabel("utility")
    out.append(f"edge {e}: "+", ".join(f"{r}:{u}" for r,u in zip(rs,us)))
fig.tight_layout();fig.savefig("part3.png",dpi=110)
open("results.txt","w").write("\n".join(out));print("\n".join(out))
