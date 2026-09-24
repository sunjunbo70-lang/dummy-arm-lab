"""Verify binary STL edge manifoldness, degeneracy, volume, and bed placement."""
import struct,json
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parent
out={}
for p in sorted((root/'models').glob('*.stl')):
    raw=p.read_bytes(); n=struct.unpack_from('<I',raw,80)[0]
    assert len(raw)==84+50*n
    edges=Counter(); degenerate=0; zs=[]; volume=0
    for k in range(n):
        t=struct.unpack_from('<12fH',raw,84+50*k)
        v=[tuple(round(x,5) for x in t[i:i+3]) for i in (3,6,9)]
        zs.extend(x[2] for x in v)
        if len(set(v))<3:degenerate+=1
        for a,b in ((0,1),(1,2),(2,0)): edges[tuple(sorted((v[a],v[b])))]+=1
        a,b,c=v
        volume+=(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6
    invalid=sum(i!=2 for i in edges.values())
    assert invalid==0 and degenerate==0 and abs(min(zs))<.001 and volume>0,(p.name,invalid,degenerate,min(zs),volume)
    out[p.name]={'triangles':n,'nonmanifold_edges':invalid,'degenerate_triangles':degenerate,'zmin_mm':min(zs),'signed_volume_mm3':volume}
(root/'mesh_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
