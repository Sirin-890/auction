import numpy as np, json
rng=np.random.default_rng(3); N=100000
grid=np.round(np.arange(0,1.0001,0.01),2)
strats={"truthful":lambda v:v,"half":lambda v:v/2,"random":lambda v:rng.random(v.shape)}
D={"p1":{}, "p2":{}}
for nr in (1,2):
    for v in (0.2,0.5,0.8):
        for sn,f in strats.items():
            u=[]
            for b in grid:
                rb=f(rng.random((N,nr))).max(1); u.append(round(float(np.where(b>rb,v-rb,0).mean()),5))
            D["p1"][f"{nr}|{v}|{sn}"]=u
ks=np.round(np.arange(0.1,2.0001,0.1),1)
for n in (2,3,4):
    us=[];fr=[]
    for k in ks:
        v=rng.random((N,n)); b=v*(n-1)/n; b[:,0]=k*v[:,0]; s=np.sort(b,1)
        us.append(round(float(np.where(b.argmax(1)==0,v[:,0]-s[:,-1],0).mean()),5)); fr.append(float((s[:,-1]-s[:,-2]>1e-12).mean()))
    v=rng.random((N,n)); b=v*(n-1)/n; s=np.sort(b,1)
    D["p2"][str(n)]={"u":us,"frac":float(np.mean(fr)),"over":round(float((s[:,-1]-s[:,-2]).mean()),4),"theory":round((n-1)/n/(n+1),4)}
D["ks"]=[float(k) for k in ks]
tpl=open("template.html").read()
open("docs/index.html","w").write(tpl.replace("__DATA__",json.dumps(D,separators=(',',':'))))
print("ok")
