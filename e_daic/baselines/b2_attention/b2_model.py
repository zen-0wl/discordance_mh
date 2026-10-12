from __future__ import annotations
import torch
import torch.nn as nn

class BaselineAttention(nn.Module):
    """Classical bidirectional cross-attention baseline; no quantum operations."""
    def __init__(self,input_dim=768,hidden_dim=256,num_heads=4,dropout=0.2,num_classes=2):
        super().__init__()
        self.audio_proj=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU())
        self.text_proj=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU())
        self.a2t=nn.MultiheadAttention(hidden_dim,num_heads,dropout=dropout,batch_first=True)
        self.t2a=nn.MultiheadAttention(hidden_dim,num_heads,dropout=dropout,batch_first=True)
        self.pool_a=nn.Linear(hidden_dim,1); self.pool_t=nn.Linear(hidden_dim,1)
        self.fusion=nn.Sequential(nn.Linear(hidden_dim*4,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(dropout))
        self.classifier=nn.Linear(256,num_classes)
    @staticmethod
    def pool(x,mask,layer):
        z=layer(x).squeeze(-1).masked_fill(~mask,torch.finfo(x.dtype).min)
        w=torch.softmax(z,1).unsqueeze(-1); return (w*x).sum(1)
    def forward(self,a,t,am=None,tm=None):
        if am is None: am=torch.ones(a.shape[:2],dtype=torch.bool,device=a.device)
        if tm is None: tm=torch.ones(t.shape[:2],dtype=torch.bool,device=t.device)
        a=self.audio_proj(a); t=self.text_proj(t)
        a2,_=self.a2t(a,t,t,key_padding_mask=~tm,need_weights=False)
        t2,_=self.t2a(t,a,a,key_padding_mask=~am,need_weights=False)
        pa=self.pool(a2,am,self.pool_a); pt=self.pool(t2,tm,self.pool_t)
        fused=torch.cat([pa,pt,torch.abs(pa-pt),pa*pt],1)
        fused=self.fusion(fused)
        return {"logits":self.classifier(fused),"pooled_audio":pa,"pooled_text":pt}
if __name__=="__main__":
    torch.manual_seed(42); m=BaselineAttention()
    a=torch.randn(2,40,768); t=torch.randn(2,15,768)
    o=m(a,t,torch.ones(2,40,dtype=torch.bool),torch.ones(2,15,dtype=torch.bool))
    print("B2 Attention forward test:",tuple(o["logits"].shape)); print("B2 model test: PASS")
