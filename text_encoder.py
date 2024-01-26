import torch
from torch import nn
from transformers import CLIPTextModel

class Embed(nn.Module):
    def __init__(self, embed_dim=768, n_tokens=77, seq_len=49408):
        super().__init__()
        self.embed = nn.Embedding(seq_len, embed_dim)
        self.pos_embed = nn.Embedding(n_tokens, embed_dim)
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.register_buffer('pos_ids', torch.arange(n_tokens).unsqueeze(0))
    def forward(self, input_ids):
        # input_ids: (b, 77)
        embed = self.embed(input_ids)
        pos_embed = self.pos_embed(self.pos_ids)
        return embed + pos_embed

class SelfAttention(nn.Module):
    def __init__(self, emb_dim=768, heads=12):
        super().__init__()
        self.wq = nn.Linear(emb_dim, emb_dim)
        self.wk = nn.Linear(emb_dim, emb_dim)
        self.wv = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.emb_dim = emb_dim
        self.heads = heads
    def get_mask(self, b, n_tok):
        mask = torch.empty(b, n_tok, n_tok)
        mask.fill_(-float('inf'))
        mask.triu_(1).unsqueeze(1)
        return mask
    def forward(self, x):
        # (b, 77, 768)
        b, n_tok, _ = x.shape
        q = self.wq(x)/8
        k = self.wk(x)
        v = self.wv(x)
        # 注意力头拆分
        q = q.reshape(b, n_tok, self.heads, self.emb_dim//self.heads).transpose(1,2).reshape(b*self.heads, n_tok, self.emb_dim//self.heads)
        k = k.reshape(b, n_tok, self.heads, self.emb_dim//self.heads).transpose(1,2).reshape(b*self.heads, n_tok, self.emb_dim//self.heads)
        v = v.reshape(b, n_tok, self.heads, self.emb_dim//self.heads).transpose(1,2).reshape(b*self.heads, n_tok, self.emb_dim//self.heads)
        # 计算q,k乘积, qk关系矩阵
        atten = torch.bmm(q, k.transpose(1,2))
        atten = atten.reshape(b, self.heads, n_tok, n_tok)
        atten = atten + self.get_mask(b, n_tok).to(atten.device)
        atten = atten.reshape(b*self.heads, n_tok, n_tok)
        atten = atten.softmax(dim=-1)
        atten = torch.bmm(atten, v) # (b*12, 77, 77)
        atten = atten.reshape(b, self.heads, n_tok, self.emb_dim//self.heads).transpose(1,2).reshape(b, n_tok, self.emb_dim) # (b, 77, 768)
        out = self.out_proj(atten)
        return out

class QuickGELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x*(x*1.702).sigmoid()

class Block(nn.Module):
    def __init__(self, embed_dim=768, expand_dim=3072):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            SelfAttention())
        self.seq2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, expand_dim),
            QuickGELU(),
            nn.Linear(expand_dim, embed_dim))
    def forward(self, x):
        x = x + self.seq1(x)
        x = x + self.seq2(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        seq = [Embed()] + [Block() for _ in range(12)] + [nn.LayerNorm(embed_dim)]
        self.seq = nn.Sequential(*seq)
    def forward(self, x):
        out = self.seq(x)
        return out

def load_pretrained(model):
    params = CLIPTextModel.from_pretrained('./pretrained-params', subfolder='text_encoder')
    model.seq[0].embed.load_state_dict(
        params.text_model.embeddings.token_embedding.state_dict())
    model.seq[0].pos_embed.load_state_dict(
        params.text_model.embeddings.position_embedding.state_dict())

    for i in range(12):
        model.seq[i+1].seq1[0].load_state_dict(
            params.text_model.encoder.layers[i].layer_norm1.state_dict())
        model.seq[i+1].seq1[1].wq.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.q_proj.state_dict())
        model.seq[i+1].seq1[1].wk.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.k_proj.state_dict())
        model.seq[i+1].seq1[1].wv.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.v_proj.state_dict())
        model.seq[i+1].seq1[1].out_proj.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.out_proj.state_dict())
        model.seq[i+1].seq2[0].load_state_dict(
            params.text_model.encoder.layers[i].layer_norm2.state_dict())
        model.seq[i+1].seq2[1].load_state_dict(
            params.text_model.encoder.layers[i].mlp.fc1.state_dict())
        model.seq[i+1].seq2[3].load_state_dict(
            params.text_model.encoder.layers[i].mlp.fc2.state_dict())
    
    model.seq[13].load_state_dict(params.text_model.final_layer_norm.state_dict())
    return model

def text_encoder_pretrained():
    text_encoder = TextEncoder()
    text_encoder = load_pretrained(text_encoder)
    return text_encoder