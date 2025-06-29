import torch
import math
import torch.nn as nn

""""
Heavily inspired by:
https://github.com/JamesQFreeman/Sam_LoRA/blob/main/sam_lora.py
https://github.com/MathieuNlp/Sam_LoRA/blob/main/src/lora.py

Slightly modified to go from shape of SAM to that of TinySAM
"""""
class LoRA_qkv(nn.Module):
    def __init__(
        self,
        base_qkv: nn.Linear,
        linear_a_q: nn.Linear,
        linear_b_q: nn.Linear,
        linear_a_v: nn.Linear,
        linear_b_v: nn.Linear,
        nh_kd: int,
        dh: int,
        rank: int = 16,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.base_qkv = base_qkv
        # Double ensure layer is frozen
        self.base_qkv.requires_grad_(False)

        # Low-rank adapters
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.nh_kd = nh_kd # dim of qk
        self.dh = dh # dim of v
        self.rank = rank
        self.scaling = alpha/self.rank



    def forward(self, x: torch.Tensor):
        # base QKV
        qkv = self.base_qkv(x)  # [B, N, 2*nh_kd + dh]
        # split
        q = qkv[..., :self.nh_kd]
        k = qkv[..., self.nh_kd:2*self.nh_kd]
        v = qkv[..., 2*self.nh_kd:]
        # LoRA updates
        q = q + self.linear_b_q(self.linear_a_q(x)) * self.scaling
        v = v + self.linear_b_v(self.linear_a_v(x)) * self.scaling
        # concat and return
        return torch.cat([q, k, v], dim=-1)


# Wrapper class for linear layers in mask decoder MLP
class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        # Double ensure layer is frozen
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.a = nn.Linear(base.in_features, self.rank, bias = False)
        self.b = nn.Linear(self.rank, base.out_features, bias = False)
        self.scaling = alpha/self.rank

      
    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scaling


# Wrapper to inject LoRA into TinyViT image encoder inside SAM
class LoRATinySAM(nn.Module):
    def __init__(
        self,
        sam_model: nn.Module,
        rank: int = 16,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.sam = sam_model
        self.rank = rank
        self.alpha = alpha

        # freeze entire SAM model
        for p in self.sam.parameters():
            p.requires_grad = False

        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()

        # Lora into image encoder
        for layer in self.sam.image_encoder.layers[1:]:  # skip initial conv stage
            for blk in layer.blocks:
                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features
                nh_kd = blk.attn.nh_kd
                dh = blk.attn.dh

                a_q = nn.Linear(self.dim, self.rank, bias=False)
                b_q = nn.Linear(self.rank, nh_kd, bias=False)
                a_v = nn.Linear(self.dim, self.rank, bias=False)
                b_v = nn.Linear(self.rank, dh, bias=False)
                self.w_As.append(a_q)
                self.w_Bs.append(b_q)
                self.w_As.append(a_v)
                self.w_Bs.append(b_v)

                blk.attn.qkv = LoRA_qkv(
                    base_qkv=w_qkv_linear,
                    linear_a_q=a_q,
                    linear_b_q=b_q,
                    linear_a_v=a_v,
                    linear_b_v=b_v,
                    nh_kd=nh_kd,
                    dh=dh,
                    rank=self.rank,
                    alpha=self.alpha,
                )
        self.lin_weights_decoder = nn.ModuleList()
        # Lora into mask decoder
        for name, module in list(self.sam.mask_decoder.named_modules()):
            if isinstance(module, nn.Linear):
                parent = self.sam.mask_decoder
                *path, attr = name.split('.')
                for p in path:
                    parent = getattr(parent, p)
                lora_lin = LoraLinear(module, self.rank, self.alpha)
                self.lin_weights_decoder.append(lora_lin)

                setattr(parent, attr, lora_lin)


        self.reset_parameters()

    def reset_parameters(self):
        # encoder
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
        # decoder
        for lin in self.lin_weights_decoder:
            if lin.b.weight.shape[1] == self.rank:
                nn.init.kaiming_uniform_(lin.a.weight, a=math.sqrt(5))
                nn.init.zeros_(lin.b.weight)
        
    def forward(self, x):
        return self.sam(x)
