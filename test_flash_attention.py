import numpy as np, torch, math, time
from numpy import array, ndarray, linalg, random as npr
from torch import nn, tensor, Tensor
from matplotlib import pyplot as plt
from torch.nn import functional as F

from cuda_transformer import FloatTensor, FloatModule, AttentionFloat, FloatAttentionLayer

B = 8
E = 1024
H = 16
D = 64

attention: FloatAttentionLayer = AttentionFloat(
    input_dim=E,
    num_heads=H,
    head_dim=D,
) # type: ignore
Wq = torch.randn(E, E).cuda() / math.sqrt(E)
Wk = torch.randn(E, E).cuda() / math.sqrt(E)
Wv = torch.randn(E, E).cuda() / math.sqrt(E)
Wo = torch.randn(E, E).cuda() / math.sqrt(E)

attention.setQueryWeights(FloatTensor.fromNumpy(Wq.cpu().numpy())) # type: ignore
attention.setKeyWeights(FloatTensor.fromNumpy(Wk.cpu().numpy())) # type: ignore
attention.setValueWeights(FloatTensor.fromNumpy(Wv.cpu().numpy())) # type: ignore
attention.setOutputWeights(FloatTensor.fromNumpy(Wo.cpu().numpy())) # type: ignore

torch_times = []
my_times = []

for N in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    X = torch.randn(B, N, E).cuda()
    out = attention(FloatTensor.fromNumpy(X.cpu().numpy())) # type: ignore
    my_times_N = []
    for _ in range(10):
        X_in = FloatTensor.fromNumpy(torch.randn(B, N, E).numpy()) # type: ignore
        torch.cuda.synchronize(); torch.cuda.synchronize(); start = time.time()
        out_ = attention(X_in) # type: ignore
        torch.cuda.synchronize(); torch.cuda.synchronize(); end = time.time()
        my_times_N.append((end - start) * 1000)
    
    Q, K, V = X @ Wq.T, X @ Wk.T, X @ Wv.T
    C = torch.zeros_like(Q)
    for b in range(0, E, D):
        q = Q[:, :, b:b+D]
        k = K[:, :, b:b+D]
        v = V[:, :, b:b+D]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        C[:, :, b:b+D] = o
    O = C @ Wo.T
    assert np.allclose(O.cpu().numpy(), out.numpy(), atol=1e-4), f"Attention output mismatch for N={N}"
    torch_times_N = []
    for _ in range(10):
        X = torch.randn(B, N, E).cuda()
        torch.cuda.synchronize(); torch.cuda.synchronize(); start = time.time()
        Q, K, V = X @ Wq.T, X @ Wk.T, X @ Wv.T
        C = torch.zeros_like(Q)
        for b in range(0, E, D):
            q = Q[:, :, b:b+D]
            k = K[:, :, b:b+D]
            v = V[:, :, b:b+D]
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            C[:, :, b:b+D] = o
        O = C @ Wo.T
        torch.cuda.synchronize(); torch.cuda.synchronize(); end = time.time()
        torch_times_N.append((end - start) * 1000)
    
    torch_times.append(np.mean(torch_times_N).item())
    my_times.append(np.mean(my_times_N).item())

plt.plot([32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], torch_times, label='PyTorch')
plt.plot([32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], my_times, label='My Implementation')
plt.xlabel('Sequence Length')
plt.ylabel('Time (ms)')
plt.legend()
plt.show()
plt.savefig('attention_times.png')
plt.close()