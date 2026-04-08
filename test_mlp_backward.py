import numpy as np, torch, time
from numpy import array, ndarray, linalg, random as npr
from torch import nn, tensor, Tensor, optim

from cuda_transformer import FloatTensor, FloatModule, MLPFloat, LinearFloat, ActivationFloat, ActivationType


W1 = torch.randn(1024, 2048).cuda() / np.sqrt(1024)
W2 = torch.randn(2048, 2048).cuda() / np.sqrt(2048)
W3 = torch.randn(2048, 2048).cuda() / np.sqrt(2048)
W4 = torch.randn(2048, 512).cuda() / np.sqrt(2048)

mlp = MLPFloat([
    LinearFloat(1024, 2048),
    ActivationFloat(2048, ActivationType.ReLU),
    LinearFloat(2048, 2048),
    ActivationFloat(2048, ActivationType.ReLU),
    LinearFloat(2048, 2048),
    ActivationFloat(2048, ActivationType.ReLU),
    LinearFloat(2048, 512),
])
mlp.setParameters({
    "layer_0.weights": FloatTensor.fromNumpy(W1.T.contiguous().cpu().numpy()),
    "layer_2.weights": FloatTensor.fromNumpy(W2.T.contiguous().cpu().numpy()),
    "layer_4.weights": FloatTensor.fromNumpy(W3.T.contiguous().cpu().numpy()),
    "layer_6.weights": FloatTensor.fromNumpy(W4.T.contiguous().cpu().numpy()),
})

torch_mlp = nn.ModuleList([
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512),
]).cuda()
torch_mlp[0].weight.data = W1.T.contiguous().detach()
torch_mlp[2].weight.data = W2.T.contiguous().detach()
torch_mlp[4].weight.data = W3.T.contiguous().detach()
torch_mlp[6].weight.data = W4.T.contiguous().detach()

torch_mlp[0].bias.data.fill_(0) # type: ignore
torch_mlp[2].bias.data.fill_(0) # type: ignore
torch_mlp[4].bias.data.fill_(0) # type: ignore
torch_mlp[6].bias.data.fill_(0) # type: ignore

loss_fn = nn.MSELoss().cuda()
optimizer = optim.SGD(torch_mlp.parameters(), lr=0.01)
optimizer.zero_grad()

X = torch.randn(1024, 1024).cuda()
Y = torch.randn(1024, 512).cuda()
X_in = FloatTensor.fromNumpy(X.cpu().numpy()) # type: ignore
X.requires_grad_(True)
Y.requires_grad_(True)

mlp.zeroGrad()
out = mlp(X_in)
out_tensor = tensor(out.numpy()).cuda().requires_grad_(True).requires_grad_(True)
loss = loss_fn(out_tensor, Y)
loss.backward()
out_grad = FloatTensor.fromNumpy(out_tensor.grad.cpu().numpy()) # type: ignore
mlp.backward(X_in, out_grad)
mlp.sgdUpdate(0.01)

out_tensor = X
for layer in torch_mlp:
    out_tensor = layer(out_tensor)
loss = loss_fn(out_tensor, Y)
loss.backward()
optimizer.step()

assert np.allclose(mlp.getParameters()["layer_0.weights"].numpy(), torch_mlp[0].weight.data.detach().cpu().numpy(), atol = 1e-4), "Layer 0 weight mismatch"
assert np.allclose(mlp.getParameters()["layer_2.weights"].numpy(), torch_mlp[2].weight.data.detach().cpu().numpy(), atol = 1e-4), "Layer 1 weight mismatch"
assert np.allclose(mlp.getParameters()["layer_4.weights"].numpy(), torch_mlp[4].weight.data.detach().cpu().numpy(), atol = 1e-4), "Layer 2 weight mismatch"
assert np.allclose(mlp.getParameters()["layer_6.weights"].numpy(), torch_mlp[6].weight.data.detach().cpu().numpy(), atol = 1e-4), "Layer 3 weight mismatch"

torch.cuda.synchronize(); start = time.time()
for _ in range(100):
    mlp.zeroGrad()
    out = mlp(X_in)
    out_tensor = tensor(out.numpy()).cuda().requires_grad_(True)
    loss = loss_fn(out_tensor, Y)
    loss.backward()
    out_grad = FloatTensor.fromNumpy(out_tensor.grad.cpu().numpy()) # type: ignore
    mlp.backward(X_in, out_grad)
    mlp.sgdUpdate(0.01)
torch.cuda.synchronize(); end = time.time()
print(f"MLP backward pass time: {(end - start) * 1000:.2f} ms")

torch.cuda.synchronize(); start = time.time()
for _ in range(100):
    optimizer.zero_grad()
    out_tensor = X
    for layer in torch_mlp:
        out_tensor = layer(out_tensor)
    loss = loss_fn(out_tensor, Y)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize(); end = time.time()
print(f"PyTorch MLP backward pass time: {(end - start) * 1000:.2f} ms")