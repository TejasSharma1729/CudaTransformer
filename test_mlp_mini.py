import numpy as np, torch, time
from numpy import array, ndarray, linalg, random as npr
from torch import nn, tensor, Tensor, optim

from cuda_transformer import FloatTensor, MLPFloat, LinearFloat, ActivationFloat, CheckpointFloat, FloatMSELoss, ActivationType

mlp_checkpoint = MLPFloat([
    LinearFloat(1024, 2048),
    CheckpointFloat(),
    LinearFloat(2048, 512),
])

loss_mlp = FloatMSELoss()

X = torch.randn(1024, 1024).cuda()
Y = torch.randn(1024, 512).cuda()
X_in = FloatTensor.fromNumpy(X.cpu().numpy()) # type: ignore
Y_in = FloatTensor.fromNumpy(Y.cpu().numpy()) # type: ignore

mlp_checkpoint.zeroGrad()
out_checkpoint = mlp_checkpoint(X_in)
out_checkpoint_grad = loss_mlp.backward(out_checkpoint, Y_in)
mlp_checkpoint.backward(X_in, out_checkpoint_grad)
