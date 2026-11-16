"""Voxel and point networks (all re-implemented in PyTorch for this study)."""
import torch
import torch.nn as nn


def dws(ci,co): return nn.Sequential(nn.Conv3d(ci,ci,3,1,1,groups=ci,bias=False),nn.BatchNorm3d(ci),nn.ReLU(inplace=True),
                                     nn.Conv3d(ci,co,1,bias=False),nn.BatchNorm3d(co),nn.ReLU(inplace=True))
def std3(ci,co): return nn.Sequential(nn.Conv3d(ci,co,3,1,1,bias=False),nn.BatchNorm3d(co),nn.ReLU(inplace=True))
class TVoxNet(nn.Module):
    """Shared per-slice 3D encoder + temporal head. tmp: conv | gru | avg; conv: dws | std; early: slices stacked as input channels."""
    def __init__(s,n=16,T=8,c=(16,32,64),d=128,tmp='conv',nc=21,conv='dws',early=False,cin=1):
        super().__init__(); s.T,s.n,s.tmp,s.early,s.cin=T,n,tmp,early,cin; blk=dws if conv=='dws' else std3
        s.enc=nn.Sequential(nn.Conv3d(T*cin if early else cin,c[0],3,2,1,bias=False),nn.BatchNorm3d(c[0]),nn.ReLU(inplace=True),
                            blk(c[0],c[1]),nn.MaxPool3d(2),blk(c[1],c[2]),nn.MaxPool3d(2))
        m=((n+1)//2)//2//2; s.proj=nn.Sequential(nn.Flatten(),nn.Linear(c[2]*m**3,d),nn.BatchNorm1d(d),nn.ReLU(inplace=True))
        if early: s.head=nn.Sequential(nn.Dropout(0.3),nn.Linear(d,nc))
        elif tmp=='conv':
            s.tc=nn.Sequential(nn.Conv1d(d,d,3,1,1,bias=False),nn.BatchNorm1d(d),nn.ReLU(inplace=True),
                               nn.Conv1d(d,d,3,1,1,bias=False),nn.BatchNorm1d(d),nn.ReLU(inplace=True))
            s.head=nn.Sequential(nn.Dropout(0.3),nn.Linear(d*T,nc))
        elif tmp=='gru': s.gru=nn.GRU(d,d,batch_first=True); s.head=nn.Sequential(nn.Dropout(0.3),nn.Linear(d,nc))
        else: s.head=nn.Sequential(nn.Dropout(0.3),nn.Linear(d,nc))
    def forward(s,x):
        B=x.shape[0]
        if s.early: return s.head(s.proj(s.enc(x.reshape(B,s.T*s.cin,s.n,s.n,s.n))))
        z=s.proj(s.enc(x.reshape(B*s.T,s.cin,s.n,s.n,s.n))).view(B,s.T,-1)
        if s.tmp=='conv': return s.head(s.tc(z.transpose(1,2)).flatten(1))
        if s.tmp=='gru': return s.head(s.gru(z)[1][0])
        return s.head(z.mean(1))
class RadHARLite(nn.Module):
    """Re-implementation of the RadHAR TD-CNN + Bi-LSTM (Singh et al., mmNets 2019): time-distributed Conv3D(16)-Conv3D(32)-MaxPool-
    Conv3D(32)-Conv3D(64)-MaxPool, Flatten, Bi-LSTM(16), Dropout 0.5, Dense. Input: 8 slices x 16^3 instead of 60 frames x 10x32x32."""
    def __init__(s,n=16,T=8,nc=21):
        super().__init__(); s.T,s.n=T,n; r=nn.ReLU(inplace=True)
        s.enc=nn.Sequential(nn.Conv3d(1,16,3,1,1),r,nn.Conv3d(16,32,3,1,1),nn.ReLU(inplace=True),nn.MaxPool3d(2),
                            nn.Conv3d(32,32,3,1,1),nn.ReLU(inplace=True),nn.Conv3d(32,64,3,1,1),nn.ReLU(inplace=True),nn.MaxPool3d(2),nn.Flatten())
        s.lstm=nn.LSTM(64*(n//4)**3,16,batch_first=True,bidirectional=True); s.head=nn.Sequential(nn.Dropout(0.5),nn.Linear(32,nc))
    def forward(s,x):
        B=x.shape[0]; z=s.enc(x.reshape(B*s.T,1,s.n,s.n,s.n)).view(B,s.T,-1); h=s.lstm(z)[1][0]
        return s.head(torch.cat([h[0],h[1]],1))
class PointNet(nn.Module):
    """PointNet without T-Nets on (x, y, z, t) (Qi et al., CVPR 2017); w/h set the widths (smaller for PointNet+t-tiny)."""
    def __init__(s,cin=4,nc=21,w=(64,128,1024),h=(512,256)):
        super().__init__(); a,b,c=w; s.c=nn.Sequential(nn.Conv1d(cin,a,1),nn.BatchNorm1d(a),nn.ReLU(),nn.Conv1d(a,b,1),nn.BatchNorm1d(b),nn.ReLU(),nn.Conv1d(b,c,1),nn.BatchNorm1d(c),nn.ReLU())
        s.h=nn.Sequential(nn.Linear(c,h[0]),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h[0],h[1]),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h[1],nc))
    def forward(s,x): return s.h(s.c(x.transpose(1,2)).max(2)[0])
def knn_feat(x,k):  # EdgeConv features [x_j - x_i, x_i] over the k nearest neighbours
    d=-2*torch.matmul(x.transpose(2,1),x)+(x**2).sum(1,keepdim=True)+(x**2).sum(1,keepdim=True).transpose(2,1)
    idx=d.topk(k,dim=-1,largest=False)[1]; B,C,N=x.shape
    xt=x.transpose(1,2); nb=xt[torch.arange(B,device=x.device)[:,None,None],idx].permute(0,3,1,2)
    xc=x.unsqueeze(3).expand(B,C,N,k); return torch.cat([nb-xc,xc],1)
class DGCNNLite(nn.Module):
    """DGCNN (Wang et al., TOG 2019) with three EdgeConv layers (k=16) and a 512-d global feature."""
    def __init__(s,cin=4,nc=21,k=16):
        super().__init__(); s.k=k
        s.e1=nn.Sequential(nn.Conv2d(2*cin,64,1,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(0.2))
        s.e2=nn.Sequential(nn.Conv2d(128,64,1,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(0.2))
        s.e3=nn.Sequential(nn.Conv2d(128,128,1,bias=False),nn.BatchNorm2d(128),nn.LeakyReLU(0.2))
        s.c=nn.Sequential(nn.Conv1d(256,512,1,bias=False),nn.BatchNorm1d(512),nn.LeakyReLU(0.2))
        s.h=nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.LeakyReLU(0.2),nn.Dropout(0.5),nn.Linear(256,nc))
    def forward(s,x):
        x=x.transpose(1,2); x1=s.e1(knn_feat(x,s.k)).max(-1)[0]; x2=s.e2(knn_feat(x1,s.k)).max(-1)[0]; x3=s.e3(knn_feat(x2,s.k)).max(-1)[0]
        return s.h(s.c(torch.cat([x1,x2,x3],1)).max(2)[0])
def fps(xyz,S):
    B,N,_=xyz.shape; ar=torch.arange(B,device=xyz.device); idx=torch.zeros(B,S,dtype=torch.long,device=xyz.device)
    dist=torch.full((B,N),1e10,device=xyz.device); far=torch.zeros(B,dtype=torch.long,device=xyz.device)
    for i in range(S):
        idx[:,i]=far; d=((xyz-xyz[ar,far][:,None])**2).sum(-1); dist=torch.minimum(dist,d); far=dist.argmax(1)
    return idx
def gather(x,idx):  # x B,N,C ; idx B,... -> B,...,C
    B=x.shape[0]; return x[torch.arange(B,device=x.device).view(B,*([1]*(idx.dim()-1))),idx]
def ball(xyz,new,r,k):
    d=torch.cdist(new,xyz)**2; B,S,N=d.shape
    idx=torch.arange(N,device=xyz.device).expand(B,S,N).clone(); idx[d>r*r]=N
    idx=idx.sort(-1)[0][...,:k]; f=idx[...,:1].expand_as(idx); return torch.where(idx==N,f,idx)
def mlp2d(ch):
    L=[]
    for a,b in zip(ch[:-1],ch[1:]): L+=[nn.Conv2d(a,b,1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True)]
    return nn.Sequential(*L)
class SA(nn.Module):  # PointNet++ set abstraction (single-scale grouping); S=None groups all points
    def __init__(s,S,r,k,ch):
        super().__init__(); s.S,s.r,s.k=S,r,k; s.m=mlp2d(ch)
    def forward(s,xyz,f):  # xyz B,N,3 ; f B,N,C or None
        if s.S is None:
            g=xyz[:,None]; gf=torch.cat([g,f[:,None]],-1) if f is not None else g; new=None
        else:
            new=gather(xyz,fps(xyz,s.S)); idx=ball(xyz,new,s.r,s.k)
            g=gather(xyz,idx)-new[:,:,None]; gf=torch.cat([g,gather(f,idx)],-1) if f is not None else g
        return new,s.m(gf.permute(0,3,1,2)).max(-1)[0].transpose(1,2)   # B,S,C
class PN2SSG(nn.Module):
    """PointNet++ SSG (Qi et al., NeurIPS 2017): grouping on xyz, t as point feature."""
    def __init__(s,nc=21):
        super().__init__(); s.sa1=SA(128,0.5,32,[3+1,64,64,128]); s.sa2=SA(32,1.0,32,[3+128,128,128,256]); s.sa3=SA(None,None,None,[3+256,256,512,1024])
        s.h=nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Dropout(0.5),nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(0.5),nn.Linear(256,nc))
    def forward(s,x):
        xyz,f=x[...,:3].contiguous(),x[...,3:]; n1,f1=s.sa1(xyz,f); n2,f2=s.sa2(n1,f1); _,f3=s.sa3(n2,f2); return s.h(f3[:,0])
class P4TLite(nn.Module):
    """P4Transformer-lite (after Fan et al., CVPR 2021): 64 anchors by FPS in (x,y,z,t) space, 4-D kNN (k=16) tube embedding (MLP on [dx,dy,dz,dt], max-pool),
    + linear 4-D position embedding, 2-layer Transformer encoder (d=128, 4 heads), max+mean pooling. Deviation from P4Transformer: one
    4-D kNN embedding layer instead of per-frame point-4D convolution with temporal/spatial radii."""
    def __init__(s,nc=21,S=64,k=16,d=128,e=(64,128)):
        super().__init__(); s.S,s.k=S,k; s.emb=mlp2d([4,e[0],e[1],d]); s.pos=nn.Linear(4,d)
        s.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,2*d,0.1,batch_first=True,norm_first=True),2)
        s.h=nn.Sequential(nn.LayerNorm(2*d),nn.Dropout(0.3),nn.Linear(2*d,nc))
    def forward(s,x):
        a=gather(x,fps(x,s.S)); idx=torch.cdist(a,x).topk(s.k,-1,largest=False)[1]
        g=gather(x,idx)-a[:,:,None]; z=s.emb(g.permute(0,3,1,2)).max(-1)[0].transpose(1,2)+s.pos(a)
        z=s.tr(z); return s.h(torch.cat([z.max(1)[0],z.mean(1)],1))
class PantomimeLite(nn.Module):
    """Re-implementation of the Pantomime architecture (Palipana et al., IMWUT 2021), scaled down:
    (i) per-frame PointNet++ shared over F frames -> FC -> 2x LSTM(32); (ii) aggregated PointNet++ on all points -> FC; concat -> FC256 -> classes.
    xyz only, as in the original. Deviations: F=8 frames x 32 points (points sorted by t then split) instead of 20x512; SA widths halved."""
    def __init__(s,nc=21,F=8):
        super().__init__(); s.F=F
        s.f1=SA(16,0.5,16,[3,64,64,128]); s.f2=SA(None,None,None,[3+128,128,256,512])
        s.ffc=nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(0.5),nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(inplace=True))
        s.lstm=nn.LSTM(128,32,2,batch_first=True,dropout=0.4)
        s.a1=SA(64,0.5,32,[3,64,64,128]); s.a2=SA(16,1.0,32,[3+128,128,128,256]); s.a3=SA(None,None,None,[3+256,256,512])
        s.afc=nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(0.5),nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(inplace=True))
        s.h=nn.Sequential(nn.Linear(32+128,256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(0.5),nn.Linear(256,nc))
    def forward(s,x):
        B,N,_=x.shape; o=x[...,3].argsort(1); x=gather(x,o); xyz=x[...,:3].contiguous()
        fr=xyz.view(B*s.F,N//s.F,3); n1,f1=s.f1(fr,None); _,f2=s.f2(n1,f1)
        tseq=s.ffc(f2[:,0]).view(B,s.F,-1); ht=s.lstm(tseq)[1][0][-1]
        m1,g1=s.a1(xyz,None); m2,g2=s.a2(m1,g1); _,g3=s.a3(m2,g2); ha=s.afc(g3[:,0])
        return s.h(torch.cat([ht,ha],1))

def make(cfg, nc):
    """Build the network described by a configuration dict (see configs.py)."""
    if cfg['kind'] == 'vox':
        return TVoxNet(cfg['n'], cfg['T'], cfg['c'], cfg['d'], cfg['tmp'], nc, cfg['conv'], cfg['early'], 2 if cfg.get('cint') else 1)
    if cfg['kind'] == 'radhar':
        return RadHARLite(cfg['n'], cfg['T'], nc=nc)
    net = {'pointnet': PointNet, 'dgcnn': DGCNNLite, 'pn2': PN2SSG, 'p4t': P4TLite, 'pantomime': PantomimeLite}[cfg['net']]
    return net(nc=nc, **cfg.get('kw', {}))
