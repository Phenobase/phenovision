import os,sys,io,random
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
def m(p):
    try:
        o=os.path.getsize(p); im=Image.open(p).convert("RGB"); r={"orig":o,"n":1,"c":0}
        for q in (85,90,95):
            b=io.BytesIO(); im.save(b,"JPEG",quality=q); r[f"jpeg{q}"]=b.tell()
        for q in (85,90):
            b=io.BytesIO(); im.save(b,"WEBP",quality=q,method=6); r[f"webp{q}"]=b.tell()
        return r
    except Exception:
        return {"orig":0,"n":1,"c":1,"jpeg85":0,"jpeg90":0,"jpeg95":0,"webp85":0,"webp90":0}
files=[]
for d in sys.argv[1:]:
    fs=[os.path.join(d,f) for f in os.listdir(d) if not f.startswith(".")]
    random.seed(0); files+=random.sample(fs,min(8000,len(fs)))
tot={}
with ProcessPoolExecutor(max_workers=16) as ex:
    for r in ex.map(m,files,chunksize=64):
        for k,v in r.items(): tot[k]=tot.get(k,0)+v
o=tot["orig"]; g=lambda x:100*(1-x/o)
print(f"FULL-RES, matched (no downscale)  ({tot['n']} imgs, {tot['c']} corrupt)")
print(f"  original          : {o/1e9:.3f} GB (100%)")
for q in (85,90,95): print(f"  JPEG q{q}          : {tot[f'jpeg{q}']/1e9:.3f} GB  (saves {g(tot[f'jpeg{q}']):.1f}% vs orig)")
for q in (85,90): print(f"  WebP q{q}          : {tot[f'webp{q}']/1e9:.3f} GB  (saves {g(tot[f'webp{q}']):.1f}% vs orig)")
print(f"\n  PURE FORMAT (matched quality, full-res):")
print(f"    WebP q85 vs JPEG q85: WebP {100*(1-tot['webp85']/tot['jpeg85']):.1f}% smaller")
print(f"    WebP q90 vs JPEG q90: WebP {100*(1-tot['webp90']/tot['jpeg90']):.1f}% smaller")
