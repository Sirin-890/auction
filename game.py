import numpy as np, matplotlib, imageio_ffmpeg
matplotlib.use("Agg")
matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
rng=np.random.default_rng(7)
V1=0.5; R=10; F=24
acts=[("truthful",lambda v:v),("bids half its value",lambda v:v/2),("bids a random number",lambda v:rng.random())]
rounds=[]
for ai,(an,f) in enumerate(acts):
    for r in range(R):
        v2=rng.random(); rounds.append((ai,r,v2,f(v2)))
# cumulative utilities for 3 bidder-1 policies on same draws
pol={"truthful (b=v)":1.0,"shade (b=0.6v)":0.6,"overbid (b=1.5v)":1.5}
def u(k,rb): b=k*V1; return V1-rb if b>rb else 0
cum={};
for ai in range(3):
    c={p:[0] for p in pol}
    for (a,r,v2,rb) in [x for x in rounds if x[0]==ai]:
        for p,k in pol.items(): c[p].append(c[p][-1]+u(k,rb))
    cum[ai]=c
fig=plt.figure(figsize=(14,7.5),dpi=80)
gs=fig.add_gridspec(1,2,width_ratios=[1.2,1])
def draw(k):
    rr=k//F; ph=k%F
    ai,r,v2,b2=rounds[rr]; b1=V1
    fig.clf(); ax=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1])
    ax.set_xlim(0,4); ax.set_ylim(0,1.15); ax.set_xticks([1,3]); ax.set_xticklabels(["Player 1 (you)\nvalue 0.5","Player 2 (rival)"],fontsize=13)
    ax.set_ylabel("value / bid")
    ax.set_title(f"Act {ai+1}: rival {acts[ai][0]}  -  round {r+1}/{R}",fontsize=15)
    # value bars
    ax.bar([1,3],[V1,v2],width=.5,color=['#9ec5fe','#f9b4b4'],edgecolor='k',alpha=.5 if ph>=8 else 1)
    ax.text(1,V1+.02,f"v={V1}",ha='center'); ax.text(3,v2+.02,f"v={v2:.2f}",ha='center') if ph>=2 else None
    if ph<2: ax.cla() ; ax.set_xlim(0,4); ax.set_ylim(0,1.15); ax.set_title(f"Act {ai+1}: rival {acts[ai][0]}  -  round {r+1}/{R}",fontsize=15); ax.text(2,.5,"new round: rival draws a value...",ha='center',fontsize=14)
    if ph>=8:
        t=min(1,(ph-7)/6)
        ax.bar([1],[b1*t],width=.3,color='tab:blue',edgecolor='k'); ax.bar([3],[b2*t],width=.3,color='tab:red',edgecolor='k')
        ax.text(1,b1*t+.02,f"bid {b1*t:.2f}",ha='center',weight='bold'); ax.text(3,b2*t+.02,f"bid {b2*t:.2f}",ha='center',weight='bold')
    if ph>=15:
        win=b1>b2; lo=min(b1,b2)
        ax.axhline(lo,color='green',ls='--'); ax.text(2,lo+.02,f"price = 2nd highest bid = {lo:.2f}",ha='center',color='green',fontsize=12)
        res=f"Player 1 WINS, profit = {V1}-{b2:.2f} = {V1-b2:.2f}" if win else f"Player 2 wins, Player 1 profit = 0"
        ax.text(2,1.07,res,ha='center',fontsize=15,weight='bold',color='tab:blue' if win else 'tab:red')
        ax.text(1 if win else 3,1.0,"WINNER",ha='center',fontsize=13,bbox=dict(fc='gold'))
    # cumulative panel
    done=r+(1 if ph>=15 else 0)
    for p,c in cum[ai].items():
        ax2.plot(range(done+1),c[:done+1],'o-',lw=2.5 if p.startswith('truth') else 1.5,label=p)
    ax2.set_xlim(0,R); ax2.set_ylim(min(0,min(min(c) for c in cum[ai].values()))-.1,max(max(c) for c in cum[ai].values())+.2)
    ax2.set_title("Cumulative profit of Player 1, same rounds",fontsize=13); ax2.set_xlabel("rounds played"); ax2.legend(loc='upper left'); ax2.grid(alpha=.3)
anim=FuncAnimation(fig,draw,frames=len(rounds)*F,interval=1000/20)
anim.save("two_player_game.mp4",writer=FFMpegWriter(fps=20,bitrate=2500))
print("done")
