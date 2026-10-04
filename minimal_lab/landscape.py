"""Public surrogate projection and observed path; never accesses response-table labels."""
import cv2
import numpy as np
from minimal_lab.policy import fit


def render(graph):
    panel=np.full((390,1280,3),(243,246,244),np.uint8)
    def text(s,x,y,size=.5): cv2.putText(panel,str(s),(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,(40,55,45),1,cv2.LINE_AA)
    X=np.asarray(graph.get('candidates',[]),float)
    names=graph.get('parameters',[])
    if X.ndim!=2 or X.shape[1]<2:
        text('Landscape awaiting a numeric domain',25,45);return panel
    obs=[n for n in graph['nodes'] if n['kind']=='observation' and n.get('value') is not None]
    axes=tuple(names.index(p) for p in graph.get('log_parameters',[]))
    levels=[np.unique(X[:,j]) for j in (0,1)]
    shape=(len(levels[1]),len(levels[0]))
    grid=np.zeros(shape)
    if obs:
        posterior=fit(X,obs,1 if graph['goal']=='maximize' else -1,axes)
        for j,b in enumerate(levels[1]):
            for i,a in enumerate(levels[0]):
                values=posterior.mean[(X[:,0]==a)&(X[:,1]==b)]
                grid[j,i]=np.min(values) if graph['goal']=='minimize' else np.max(values)
        lo,hi=grid.min(),grid.max()
        grey=np.uint8(255*(grid-lo)/max(hi-lo,1e-6))
        heat=cv2.applyColorMap(grey,cv2.COLORMAP_VIRIDIS)
        text(f'Predicted survival range {lo:.1f} - {hi:.1f}%; uncalibrated surrogate',420,76,.56)
    else:
        heat=np.full((*shape,3),210,np.uint8)
        text('No observations: response landscape unknown',420,76,.56)
    heat=cv2.resize(heat,(300,300),interpolation=cv2.INTER_NEAREST)
    panel[48:348,65:365]=heat[::-1]
    def point(candidate):
        a,b=X[candidate,:2]; i=int(np.argmin(abs(levels[0]-a)));j=int(np.argmin(abs(levels[1]-b)))
        return int(65+(i+.5)*300/shape[1]),int(348-(j+.5)*300/shape[0])
    path=[]
    for o in obs:
        if not path or path[-1]!=o['candidate']:path.append(o['candidate'])
    for a,b in zip(path,path[1:]):cv2.arrowedLine(panel,point(a),point(b),(245,245,245),1,tipLength=.2)
    if path:
        cv2.circle(panel,point(path[0]),7,(0,180,255),2)
        cv2.circle(panel,point(path[-1]),6,(20,20,240),-1)
    text(names[0]+' (log-dose levels)',70,377,.45)
    text(names[1]+' (vertical)',65,30,.43)
    text('OPTIMISATION LANDSCAPE + PATH',420,39,.75)
    text('Projection: best predicted value over the third drug.',420,112,.54)
    text('Not ground truth. White arrows = tested path.',420,148,.54)
    text('Orange ring = literature-seeded start; red = latest recipe.',420,184,.51)
    text(f'{len(obs)} preparations / {len(set(o["candidate"] for o in obs))} distinct recipes',420,225,.64)
    if obs:
        means={}
        trajectory=[]
        for o in obs:
            means.setdefault(o['candidate'],[]).append(o['value'])
            aggregate=[np.mean(v) for v in means.values()]
            trajectory.append(min(aggregate) if graph['goal']=='minimize' else max(aggregate))
        lo,hi=min(trajectory),max(trajectory)
        pts=np.array([(440+int(i*730/max(1,len(trajectory)-1)),325-int((y-lo)*55/max(hi-lo,1)))
                      for i,y in enumerate(trajectory)],np.int32)
        cv2.polylines(panel,[pts],False,(80,135,70),2)
        text(f'Best observed mean: {trajectory[-1]:.2f}% (includes all repeats)',420,368,.55)
    return panel
