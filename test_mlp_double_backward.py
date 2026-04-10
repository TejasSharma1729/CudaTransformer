import numpy as np, torch, time
from numpy import array, ndarray, linalg, random as npr
from torch import nn, tensor, Tensor, optim

from cuda_transformer import DoubleTensor, MLPDouble, LinearDouble, ActivationDouble, CheckpointDouble, DoubleMSELoss, ActivationType


W1 = torch.randn(1024, 2048).double().cuda() / np.sqrt(1024)
W2 = torch.randn(2048, 2048).double().cuda() / np.sqrt(2048)
W3 = torch.randn(2048, 2048).double().cuda() / np.sqrt(2048)
W4 = torch.randn(2048, 512).double().cuda() / np.sqrt(2048)

mlp = MLPDouble([
    LinearDouble(1024, 2048),
    ActivationDouble(2048, ActivationType.ReLU),
    LinearDouble(2048, 2048),
    ActivationDouble(2048, ActivationType.ReLU),
    LinearDouble(2048, 2048),
    ActivationDouble(2048, ActivationType.ReLU),
    LinearDouble(2048, 512),
])
mlp.setParameters({
    "layer_0.weights": DoubleTensor.fromNumpy(W1.T.contiguous().cpu().numpy()),
    "layer_2.weights": DoubleTensor.fromNumpy(W2.T.contiguous().cpu().numpy()),
    "layer_4.weights": DoubleTensor.fromNumpy(W3.T.contiguous().cpu().numpy()),
    "layer_6.weights": DoubleTensor.fromNumpy(W4.T.contiguous().cpu().numpy()),
})

mlp_checkpoint = MLPDouble([
    LinearDouble(1024, 2048),
    ActivationDouble(2048, ActivationType.ReLU),
    LinearDouble(2048, 2048),
    ActivationDouble(2048, ActivationType.ReLU),
    CheckpointDouble(),
    LinearDouble(2048, 2048),
    ActivationDouble(2048, ActivationType.ReLU),
    LinearDouble(2048, 512),
])
mlp_checkpoint.setParameters({
    "layer_0.weights": DoubleTensor.fromNumpy(W1.T.contiguous().cpu().numpy()),
    "layer_2.weights": DoubleTensor.fromNumpy(W2.T.contiguous().cpu().numpy()),
    "layer_5.weights": DoubleTensor.fromNumpy(W3.T.contiguous().cpu().numpy()),
    "layer_7.weights": DoubleTensor.fromNumpy(W4.T.contiguous().cpu().numpy()),
})

torch_mlp = nn.ModuleList([
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512),
]).double().cuda()
torch_mlp[0].weight.data = W1.T.contiguous().detach()
torch_mlp[2].weight.data = W2.T.contiguous().detach()
torch_mlp[4].weight.data = W3.T.contiguous().detach()
torch_mlp[6].weight.data = W4.T.contiguous().detach()

torch_mlp[0].bias.data.fill_(0) # type: ignore
torch_mlp[2].bias.data.fill_(0) # type: ignore
torch_mlp[4].bias.data.fill_(0) # type: ignore
torch_mlp[6].bias.data.fill_(0) # type: ignore

loss_fn = nn.MSELoss(reduction='mean').double().cuda()
loss_mlp = DoubleMSELoss()
optimizer = optim.SGD(torch_mlp.parameters(), lr=0.01)
optimizer.zero_grad()

X = torch.randn(1024, 1024).double().cuda()
Y = torch.randn(1024, 512).double().cuda()
X_in = DoubleTensor.fromNumpy(X.cpu().numpy()) # type: ignore
Y_in = DoubleTensor.fromNumpy(Y.cpu().numpy()) # type: ignore
X.requires_grad_(True)
Y.requires_grad_(True)

mlp.zeroGrad()
out = mlp(X_in)
out_grad = loss_mlp.backward(out, Y_in)
mlp.backward(X_in, out_grad)
mlp.sgdUpdate(0.01)

mlp_checkpoint.zeroGrad()
out_checkpoint = mlp_checkpoint(X_in)
out_checkpoint_grad = loss_mlp.backward(out_checkpoint, Y_in)
mlp_checkpoint.backward(X_in, out_checkpoint_grad)
mlp_checkpoint.sgdUpdate(0.01)

out_tensor = X
for layer in torch_mlp:
    out_tensor = layer(out_tensor)
loss = loss_fn(out_tensor, Y)
loss.backward()
optimizer.step()

assert np.allclose(mlp_checkpoint.getParameters()["layer_0.weights"].numpy(), mlp.getParameters()["layer_0.weights"].numpy(), atol = 1e-4), "Checkpoint Layer 0 weight mismatch"
assert np.allclose(mlp_checkpoint.getParameters()["layer_2.weights"].numpy(), mlp.getParameters()["layer_2.weights"].numpy(), atol = 1e-4), "Checkpoint Layer 1 weight mismatch"
assert np.allclose(mlp_checkpoint.getParameters()["layer_5.weights"].numpy(), mlp.getParameters()["layer_4.weights"].numpy(), atol = 1e-4), "Checkpoint Layer 2 weight mismatch"
assert np.allclose(mlp_checkpoint.getParameters()["layer_7.weights"].numpy(), mlp.getParameters()["layer_6.weights"].numpy(), atol = 1e-4), "Checkpoint Layer 3 weight mismatch"

assert np.allclose(mlp.getParameters()["layer_0.weights"].numpy(), torch_mlp[0].weight.data.detach().cpu().numpy(), atol = 1e-4), "Layer 0 weight mismatch"
assert np.allclose(mlp.getParameters()["layer_2.weights"].numpy(), torch_mlp[2].weight.data.detach().cpu().numpy(), atol = 1e-4), "Layer 1 weight mismatch"
assert np.allclose(mlp.getParameters()["layer_4.weights"].numpy(), torch_mlp[4].weight.data.detach().cpu().numpy(), atol = 1e-4), "Layer 2 weight mismatch"
assert np.allclose(mlp.getParameters()["layer_6.weights"].numpy(), torch_mlp[6].weight.data.detach().cpu().numpy(), atol = 1e-4), "Layer 3 weight mismatch"

torch.cuda.synchronize(); start = time.time()
for _ in range(100):
    mlp.zeroGrad()
    out = mlp(X_in)
    out_grad = loss_mlp.backward(out, Y_in)
    mlp.backward(X_in, out_grad)
    mlp.sgdUpdate(0.01)
torch.cuda.synchronize(); end = time.time()
print(f"MLP backward pass time: {(end - start) * 1000:.2f} ms")

torch.cuda.synchronize(); start = time.time()
for _ in range(100):
    mlp_checkpoint.zeroGrad()
    out_checkpoint = mlp_checkpoint(X_in)
    out_checkpoint_grad = loss_mlp.backward(out_checkpoint, Y_in)
    mlp_checkpoint.backward(X_in, out_checkpoint_grad)
    mlp_checkpoint.sgdUpdate(0.01)
torch.cuda.synchronize(); end = time.time()
print(f"MLP with checkpointing backward pass time: {(end - start) * 1000:.2f} ms")

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