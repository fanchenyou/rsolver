# %Riemannian subgradient methods with geometrically diminishing stepsize
import torch
import torch.nn.functional as F

import random
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

# close all; clear all;
# % randn('seed',2019);rand('seed',2019)
# % rng('default');
# % rng('shuffle');


# %% setup the data
# D = 100; %ambient dimension
# N = 15*D; % number of inliers
# ratio = 1 ./ (1 ./ 0.7 - 1); % outlier ratio
# M = floor(N * ratio); % number of outliers
# d = 0.9*D; % subspace dimension
# c = D -d;
# X = [normc( randn(d,N) );zeros(D-d,N)];
# O = normc(randn(D,M));
# Ytilde = [X O];
# obj = @(B) sum(sqrt(sum((B'*Ytilde).^2,1)));
# % initialization
# % [Bo,~] = eigs(Ytilde*Ytilde',c,'SM');
# Bo = randn(D,c);
#

# setup the data
D = 100  # ambient dimension
N = 15 * D  # number of inliers
ratio = 1 / (1 / 0.7 - 1)  # outlier ratio
M = math.floor(N * ratio)  # number of outliers
d = int(0.9 * D)  # subspace dimension
c = D - d

mu_0 = 0.1
beta = .55
Niter = 100
dist3 = []

# pytorch
X = torch.zeros(D, N)
tmp = torch.randn(d, N)
tmp = F.normalize(tmp, p=2, dim=0)
X[:d, :] = tmp
O = F.normalize(torch.randn(D, M), p=2, dim=0)
Ytilde = torch.cat([X, O], dim=1)
print(Ytilde.size())
B = torch.randn(D, c)
# Ytilde = torch.from_numpy(Ytilde).float()

for i in range(1, Niter + 1):
    normB = torch.norm(B[:d, :], 'fro')
    dist3.append(normB.item())
    # dist3(i) = norm(B(1:d,:),'fro');
    if i % 10 == 0:
        print(i)
    mu = mu_0 * (beta ** i)

    for j in range(1, M + N + 1):
        # index = randperm(M+N,1)
        # index = np.random.permutation(M+N)[0]
        index = np.random.randint(0, M + N)

        # Y = Ytilde(:,index);
        Y = Ytilde[:, index:index + 1]
        # BY = B'*Y;
        BY = B.transpose(0, 1).mm(Y)
        # print(B.shape, Y.shape)

        # temp = norm(BY)
        # %  temp = sqrt(sum((B'*Ytilde(:,index)).^2));
        temp = torch.norm(BY)

        if temp == 0:
            continue
        else:
            # grad = Y*(BY'/temp);
            # print(Y.shape, BY.shape)
            grad = Y.mm(torch.t(BY) / temp)
            # print(Y.shape, BY.T.shape)
            # gradB = grad'*B;
            gradB = torch.t(grad).mm(B)

            # grad = grad - 0.5*B*(gradB+ gradB');
            # %  grad = grad - 0.5*B*(grad'*B+ B'*grad);

            grad = grad - 0.5 * B.mm(gradB + torch.t(gradB))  ################# projection

            #
            # B_plus = B - mu*grad;

            B_plus = B - mu * grad
            # %polar retraction
            # % B_power = B_plus'*B_plus;
            # %[U,Sigma] = eig(B_power);
            # % SIGMA =diag(Sigma);
            # %B = B_plus*(U*diag(sqrt(1./SIGMA))*U');
            #
            # %qr retraction
            # [B,~] = qr(B_plus,0);

            B, _ = torch.qr(B_plus)




print(dist3, len(dist3))
plt.semilogy(list(range(len(dist3))), dist3)
plt.show()
