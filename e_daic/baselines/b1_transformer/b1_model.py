from __future__ import annotations
import torch
import torch.nn as nn

class BaselineTransformer(nn.Module):
    """Classical transformer-fusion baseline; no quantum operations."""
    def __init__(self, input_dim=768, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.2, num_classes=2):
        super().__init__()
        self.audio_proj=nn.Linear(input_dim,hidden_dim)
        self.text_proj=nn.Linear(input_dim,hidden_dim)
        self.audio_type=nn.Parameter(torch.zeros(1,1,hidden_dim))
        self.text_type=nn.Parameter(torch.zeros(1,1,hidden_dim))
        layer=nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=num_heads,dim_feedforward=hidden_dim*4,dropout=dropout,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=num_layers)
        self.norm=nn.LayerNorm(hidden_dim)
        self.classifier=nn.Sequential(nn.Linear(hidden_dim,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,num_classes))
    @staticmethod
    def masked_mean(x,mask):
        w=mask.unsqueeze(-1).to(x.dtype); return (x*w).sum(1)/w.sum(1).clamp_min(1)
    def forward(self,a,t,am=None,tm=None):
        if am is None: am=torch.ones(a.shape[:2],dtype=torch.bool,device=a.device)
        if tm is None: tm=torch.ones(t.shape[:2],dtype=torch.bool,device=t.device)
        a=self.audio_proj(a)+self.audio_type; t=self.text_proj(t)+self.text_type
        x=torch.cat([a,t],1); mask=torch.cat([am,tm],1)
        x=self.encoder(x,src_key_padding_mask=~mask)
        pooled=self.norm(self.masked_mean(x,mask))
        return {"logits":self.classifier(pooled),"pooled":pooled}
if __name__=="__main__":
    torch.manual_seed(42); m=BaselineTransformer()
    a=torch.randn(2,40,768); t=torch.randn(2,15,768)
    o=m(a,t,torch.ones(2,40,dtype=torch.bool),torch.ones(2,15,dtype=torch.bool))
    print("B1 Transformer forward test:",tuple(o["logits"].shape)); print("B1 model test: PASS")
