import numpy as np, matplotlib, imageio_ffmpeg
matplotlib.use("Agg")
matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
rng=np.random.default_rng(1); N=100000
grid=np.round(np.arange(0,1.0001,0.01),2)
strats={"truthful":lambda v:v,"half value":lambda v:v/2,"random":lambda v:rng.random(v.shape)}
def p1(nr):
    R={}
    for v in (0.2,0.5,0.8):
        for sn,f in strats.items():
            u=[]
            for b in grid:
                rb=f(rng.random((N,nr))).max(1); u.append(np.where(b>rb,v-rb,0).mean())
            R[v,sn]=np.array(u)
    return R
D1,D2=p1(1),p1(2)
ks=np.round(np.arange(0.1,2.0001,0.1),1); D3={}
for n in (2,3,4):
    u=[]
    for k in ks:
        v=rng.random((N,n)); b=v*(n-1)/n; b[:,0]=k*v[:,0]; t=b.max(1)
        u.append(np.where(b.argmax(1)==0,v[:,0]-t,0).mean())
    D3[n]=np.array(u)
fig=plt.figure(figsize=(14,8),dpi=80)
FPS=20; scenes=[]
def scene(frames):
    def d(f): scenes.append((f,frames)); return f
    return d
def title(text,sub,n=50):
    def s(i):
        fig.clf(); fig.text(.5,.58,text,ha='center',fontsize=30,weight='bold'); fig.text(.5,.45,sub,ha='center',fontsize=17)
    scenes.append((s,n))
def grid_scene(D,label,n=110):
    def s(i):
        if i==0:
            fig.clf(); s.ax=fig.subplots(3,3,sharey='row'); fig.suptitle(label,fontsize=18)
        m=int(len(grid)*min(1,(i+1)/(n*0.8)))+0
        for a,((v,sn),u) in zip(s.ax.flat,D.items()):
            a.clear(); a.plot(grid[:m],u[:m],lw=2); a.axvline(v,color='r',ls='--'); a.set_xlim(0,1); a.set_ylim(0,u.max()*1.15)
            a.set_title(f"v={v}, rival: {sn}",fontsize=10)
            if m>1:
                j=u[:m].argmax(); a.plot(grid[j],u[j],'go')
        for a in s.ax[-1]: a.set_xlabel("bid of bidder 1")
        for a in s.ax[:,0]: a.set_ylabel("avg utility")
        fig.text(.5,.01,"red dashed = truthful bid b=v, green dot = best bid so far",ha='center') if i==0 else None
    scenes.append((s,n))
def fixed_scene(n=90):
    def s(i):
        if i==0:
            fig.clf(); s.ax=fig.subplots(1,3,sharey=True); fig.suptitle("Part 1B: fixed rival bid, bidder 1 value v=0.5 - truthful bid is always optimal",fontsize=16)
        m=int(len(grid)*min(1,(i+1)/(n*0.8)))
        for a,r in zip(s.ax,(0.1,0.4,0.9)):
            a.clear(); u=np.array([max(0.5-r,0) if b>r else 0 for b in grid])
            a.plot(grid[:m],u[:m],lw=3); a.axvline(0.5,color='r',ls='--'); a.axvline(r,color='gray',ls=':')
            a.set_xlim(0,1);a.set_ylim(-.02,.75);a.set_title(f"rival bid {r} (gray); b=v=0.5 (red)");a.set_xlabel("bid")
    scenes.append((s,n))
def p2_scene(n=120):
    def s(i):
        if i==0:
            fig.clf(); s.ax=fig.subplots(1,3,sharey=True); fig.suptitle("Part 2: first-price auction - bidder 1 bids k*v, others bid (n-1)/n*v",fontsize=16)
        m=max(1,int(len(ks)*min(1,(i+1)/(n*0.8))))
        for a,nn in zip(s.ax,(2,3,4)):
            a.clear(); a.plot(ks[:m],D3[nn][:m],'o-'); a.axvline((nn-1)/nn,color='r',ls='--'); a.set_xlim(0,2.05); a.set_ylim(0,.3)
            a.set_title(f"n={nn}; red: equilibrium k={(nn-1)/nn:.2f}"); a.set_xlabel("k")
        s.ax[0].set_ylabel("expected utility")
    scenes.append((s,n))
pos={"A":(0,.5),"B":(1,1),"C":(1,0),"D":(2,1),"E":(2,0),"F":(3,.5)}
E={"AB":3,"AC":2,"BD":2,"BE":1,"CF":5,"CE":3,"DF":2,"EF":1}
def vcg_scene(n=90):
    onp=["AB","BE","EF"]; pay={"AB":4,"BE":2,"EF":3}
    def s(i):
        fig.clf(); ax=fig.add_subplot(111); ax.axis('off'); ax.set_xlim(-.3,3.3); ax.set_ylim(-.3,1.3)
        ax.set_title("Part 3: VCG cheapest path A->F. Chosen path ABEF (cost 5); payments AB=4, BE=2, EF=3",fontsize=15)
        show=i/n>0.3
        for e,c in E.items():
            (x1,y1),(x2,y2)=pos[e[0]],pos[e[1]]; hl=show and e in onp
            ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='->',lw=5 if hl else 1.5,color='crimson' if hl else 'gray',shrinkA=22,shrinkB=22))
            t=f"cost {c}"+(f"\npaid {pay[e]}" if hl and i/n>0.6 else (" \npaid 0" if i/n>0.6 else ""))
            ax.text((x1+x2)/2,(y1+y2)/2+.04,t,ha='center',fontsize=11,bbox=dict(fc='white',ec='none',alpha=.8))
        for k,(x,y) in pos.items(): ax.text(x,y,k,ha='center',va='center',fontsize=18,bbox=dict(boxstyle='circle',fc='lightyellow'))
    scenes.append((s,n))
def vcg_u_scene(n=100):
    rs=np.arange(0,10.01,.5)
    U={"BE":[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"CE":[-1]*4+[0]*17}
    def s(i):
        if i==0:
            fig.clf(); s.ax=fig.subplots(1,2,sharey=True); fig.suptitle("Part 3: utility vs reported cost - truth-telling is never beaten",fontsize=16)
        m=max(1,int(len(rs)*min(1,(i+1)/(n*0.8))))
        for a,(e,tc,lab) in zip(s.ax,(("BE",1,"on path"),("CE",3,"off path"))):
            a.clear(); a.step(rs[:m],U[e][:m],where='post',lw=3); a.axvline(tc,color='r',ls='--'); a.set_xlim(0,10);a.set_ylim(-1.5,1.5)
            a.set_title(f"edge {e} ({lab}), true cost {tc}"); a.set_xlabel("reported cost")
        s.ax[0].set_ylabel("utility")
    scenes.append((s,n))
title("Auctions & Mechanism Design","Second-price, first-price and VCG: simulations")
title("Part 1A: Second-price, 1 rival","Utility of bidder 1 vs its bid (100,000 auctions per bid)",30)
grid_scene(D1,"Part 1A: one rival")
title("Part 1C: Second-price, 2 rivals","Same experiment with two identical rivals",30)
grid_scene(D2,"Part 1C: two rivals")
fixed_scene()
title("Part 2: First-price auction","n = 2, 3, 4 bidders",30)
p2_scene()
title("Part 3: VCG shortest path","Figure 7.3 - cheapest path A to F",30)
vcg_scene(); vcg_u_scene()
total=sum(n for _,n in scenes)
idx=[(s,i) for s,n in scenes for i in range(n)]
# hold last frame
def upd(k):
    s,i=idx[k]; s(i)
anim=FuncAnimation(fig,upd,frames=len(idx),interval=1000/FPS)
anim.save("auction_visualization.mp4",writer=FFMpegWriter(fps=FPS,bitrate=2500))
print("done",len(idx)/FPS,"s")
