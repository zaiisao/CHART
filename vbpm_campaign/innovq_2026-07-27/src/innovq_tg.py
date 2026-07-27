"""InnovQ + tempogram-fed init head.

The log-tempo LEVEL mu_l1 was read off a mean-pooled GRU state (measured 15.3% MAE).
A tempogram over the SAME 2 channels reaches 1.5% once the octave is resolved. The
tempogram is a deterministic differentiable function of h, so feeding it to the posterior
keeps q = q(z | h, b): same evidence, better-shaped statistic. No new inputs.

Faithfulness ledger: no change to the generative model. Amortization statistic only.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
import innovq as IQ

LO, HI = 10, 150
TG_DIM = 2 * (HI - LO)


def torch_tgram(h, lo=LO, hi=HI):
    """Differentiable normalized ACF (tempogram) per channel. h [B,T,C] -> [B, C*(hi-lo)]."""
    B, T, C = h.shape
    x = h - h.mean(1, keepdim=True)
    n = 1 << int(math.ceil(math.log2(2 * T)))
    f = torch.fft.rfft(x, n=n, dim=1)
    a = torch.fft.irfft(f * f.conj(), n=n, dim=1)[:, : hi + 2]          # [B,hi+2,C]
    a = a / (a[:, :1] + 1e-9)
    a = F.avg_pool1d(a.transpose(1, 2), 3, stride=1, padding=1).transpose(1, 2)
    return a[:, lo:hi].transpose(1, 2).reshape(B, -1)


class _TGInit(nn.Module):
    """init_head that also sees the cached tempogram embedding (set by encode_posterior)."""

    def __init__(self, owner, base_in, hid, out, tg_hid):
        super().__init__()
        self._owner = [owner]                       # list -> not a submodule, no recursion
        self.net = nn.Sequential(nn.Linear(base_in + tg_hid, hid), nn.Tanh(),
                                 nn.Linear(hid, out))

    def __getitem__(self, i):                       # keep InnovQ's init_head[-1] bias surgery
        return self.net[i]

    def forward(self, x):
        tg = self._owner[0]._tg
        return self.net(torch.cat([x, tg], -1))


class InnovQT(IQ.InnovQ):
    def __init__(self, tg_hid=128, **kw):
        super().__init__(**kw)
        hid = self.hidden
        self.tg_embed = nn.Sequential(nn.Linear(TG_DIM, 256), nn.GELU(),
                                      nn.Linear(256, tg_hid), nn.GELU())
        self.init_head = _TGInit(self, 2 * hid, hid, self.K + 5, tg_hid)
        with torch.no_grad():                       # redo InnovQ's init on the new head
            self.init_head[-1].weight.normal_(0.0, 1e-3)
            self.init_head[-1].bias.zero_()
            self.init_head[-1].bias[self.K] = 1.0
            self.init_head[-1].bias[self.K + 4] = IQ.B_LS0
        self._tg = None

    def encode_posterior(self, h, b):
        self._tg = self.tg_embed(torch_tgram(h))    # cache for init_head
        return super().encode_posterior(h, b)
