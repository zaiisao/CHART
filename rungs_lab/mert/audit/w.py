import torch, numpy as np
c=torch.load("/home/sogang/jaehoon/VBPM/rungs_lab/mert/runs/mertr4_mertfull_bestsel.pt",weights_only=False,map_location="cpu")
sd=c["model"]; W=sd["trunk.embed.weight"]  # [128, 3328]
print("W",W.shape,"selected_step",c["selected_step"],c["selected_nll"])
bt=W[:,:256]; me=W[:,256:]
print("BT  block: fro %.4f  per-dim rms %.5f"%(bt.norm(), bt.pow(2).mean().sqrt()))
print("MERT block: fro %.4f  per-dim rms %.5f"%(me.norm(), me.pow(2).mean().sqrt()))
print("BT frac of squared norm: %.4f (dim frac %.4f)"%(bt.pow(2).sum()/W.pow(2).sum(), 256/3328))
# per-layer chunks of MERT
for i in range(4):
    b=me[:,i*768:(i+1)*768]; print("  mert layer chunk",i,"rms %.5f"%b.pow(2).mean().sqrt())
print("init rms would be", (1/np.sqrt(3328))/np.sqrt(3))
for k in ("prior_head.weight","kernel_head.weight"):
    print(k, sd[k].norm().item())
