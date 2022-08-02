import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform
# Imports for plotting
import numpy as np
import matplotlib.pyplot as plt
# Define grids of points (for later plots)
x = np.linspace(-4, 4, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])


class Flow(transform.Transform, nn.Module):

    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)

    # Init all parameters
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)

class PlanarFlow(Flow):

    def __init__(self, dim, h=torch.tanh, hp=(lambda x: 1 - torch.tanh(x) ** 2)):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.h = h
        self.hp = hp #h的导数
        self.init_parameters()

    def _call(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.h(f_z)

    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        psi = self.hp(f_z) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-9)

def change_density(q0_density, flow, z):
    # Apply our transform on coordinates 在坐标上应用我们的变换
    f_z = flow(torch.Tensor(z)).detach() # detach 分离
    # Obtain our density
    q1_density = q0_density.squeeze() / np.exp(flow.log_abs_det_jacobian(torch.Tensor(z)).detach().squeeze())
    return q1_density, f_z

# Our base density
q0 = distrib.MultivariateNormal(torch.zeros(2), torch.eye(2))
q0_density = torch.exp(q0.log_prob(torch.Tensor(z)))
# Our transform
flow1 = PlanarFlow(2)
# Manually set the transform parameters (I know it is dirty ^^)手动设置transform参数
flow1.weight.data = torch.Tensor([[1, 0.1]])
flow1.scale.data = torch.Tensor([[0.5, 0]])
flow1.bias.data = torch.Tensor([0])
q1_density, f_z1 = change_density(q0_density, flow1, z)

flow2 = PlanarFlow(2)
flow2.weight.data = torch.Tensor([[1, 0.3]])
flow2.scale.data = torch.Tensor([[1, 0]])
flow2.bias.data = torch.Tensor([0])
q2_density, f_z2 = change_density(q1_density, flow2, f_z1)

flow3 = PlanarFlow(2)
flow3.weight.data = torch.Tensor([[1, 0.5]])
flow3.scale.data = torch.Tensor([[1.5, 0]])
flow3.bias.data = torch.Tensor([0])
q3_density, f_z3 = change_density(q2_density, flow3, f_z2)

flow4 = PlanarFlow(2)
flow4.weight.data = torch.Tensor([[1, 0.2]])
flow4.scale.data = torch.Tensor([[0.5, 0]])
flow4.bias.data = torch.Tensor([0])
q4_density, f_z4 = change_density(q3_density, flow4, f_z3)

flow5 = PlanarFlow(2)
flow5.weight.data = torch.Tensor([[4, 1]])
flow5.scale.data = torch.Tensor([[2.2, 0]])
flow5.bias.data = torch.Tensor([1.5])
q5_density, f_z5 = change_density(q4_density, flow5, f_z4)

# Plot this
# fig, (ax0, ax1, ax2, ) = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

fig = plt.figure(figsize=(15, 10))
fig.add_subplot(2,3,1)
plt.hexbin(z[:,0], z[:,1], C=q0_density.numpy().squeeze(), cmap='BrBG_r')# BrBG_r
plt.title('$q_0 = \mathcal{N}(\mathbf{0},\mathbb{I})$', fontsize=18)

fig.add_subplot(2,3,2)
plt.hexbin(f_z1[:,0], f_z1[:,1], C=q1_density.numpy().squeeze(), cmap='BrBG_r')
plt.title('$q_1=planar(q_0)$', fontsize=18)

fig.add_subplot(2,3,3)
plt.hexbin(f_z2[:,0], f_z2[:,1], C=q2_density.numpy().squeeze(), cmap='BrBG_r')
plt.title('$q_2=planar(q_1)$', fontsize=18)

fig.add_subplot(2,3,4)
plt.hexbin(f_z3[:,0], f_z2[:,1], C=q3_density.numpy().squeeze(), cmap='BrBG_r')
plt.title('$q_3=planar(q_2)$', fontsize=18)

fig.add_subplot(2,3,5)
plt.hexbin(f_z4[:,0], f_z4[:,1], C=q4_density.numpy().squeeze(), cmap='BrBG_r')
plt.title('$q_4=planar(q_3)$', fontsize=18)

fig.add_subplot(2,3,6)
plt.hexbin(f_z5[:,0], f_z5[:,1], C=q4_density.numpy().squeeze(), cmap='BrBG_r')
plt.title('$q_5=planar(q_4)$', fontsize=18)
plt.show()
