from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import models.flows as flows
from torch.nn.modules import LSTM

class VAE(nn.Module):
    """
    The base VAE class containing gated convolutional encoder and decoder architecture.
    基本VAE类包含门控卷积编码器和解码器体系结构。
    Can be used as a base class for VAE's with normalizing flows.
    可以作为带有标准化流的VAE的基类。
    """

    def __init__(self, args):
        super(VAE, self).__init__()

        # extract model settings from args 从参数中提取模型设置
        self.z_size = args.z_size
        self.input_size = args.input_size
        self.input_type = args.input_type

        if self.input_size == [24,1]:
            self.last_kernel_size = 24
        else:
            raise ValueError('invalid input size!!')

        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()

        self.q_z_nn_output_dim = 24

        # auxiliary 辅助
        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        # log-det-jacobian = 0 without flows
        self.log_det_j = Variable(self.FloatTensor(1).zero_())

    def create_encoder(self):
        """
        编码器期望数据作为形状的输入(batch_size,features, steps)。
        """

        if self.input_type == 'binary':
            q_z_nn =nn.Sequential(
                LSTM(input_size=self.input_size[1], hidden_size=24,batch_first=True)) # 1,24

            q_z_mean = nn.Linear(24, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(24, self.z_size),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=1.),
            )
            return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):

        if self.input_type == 'binary':
            p_x_nn = nn.Sequential(LSTM(input_size=1,hidden_size=24,batch_first=True)) # 24 ,24

            p_x_mean = nn.Sequential(
                nn.Linear(24, 24),
                nn.Sigmoid(),
                nn.Linear(24, self.input_size[1])
            )
            return p_x_nn, p_x_mean
        else:
            raise ValueError('invalid input type!!')

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the reparameterization trick.
        样本z从一个多元高斯与对角协方差矩阵使用重新参数化技巧。
        """

        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)

        return z

    def encode(self, x):
        """
        Encoder expects following data shapes as input: shape = (batch_size, num_channels, width, height)
        编码器期望以下数据形状作为输入: shape = (batch_size, steps, feature_num)
        """

        h1,(h2,h3) = self.q_z_nn(x)
        h = h1[:, :, -1].unsqueeze(2)
        # print('en_h.shape',h.shape) # torch.Size([24, 24, 1])

        h = h.view(h.size(0), -1)
        # print('en_h2.shape', h.shape) # torch.Size([24, 24])

        mean = self.q_z_mean(h)
        # print('en_mean.shape', mean.shape) #torch.Size([24, 24])

        var = self.q_z_var(h)
        # print('en_var.shape', var.shape)#torch.Size([24, 24])

        return mean, var

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        解码器输出重构图像如下:
        x_mean.shape = (batch_size, steps, feature_num) 100 , 24 , 1
        """
        # print('解码器的输入:z的形状',z.shape) # torch.Size([100, 24])

        z = z.view(z.size(0), self.z_size,1)
        # print('de_z.shape',z.shape) # torch.Size([30, 24, 1])

        h1,(h2,h3) = self.p_x_nn(z)#100,24,1
        # print('de_h1.shape',h1.shape) #de_h1.shape torch.Size([100, 24, 24])

        x_mean = self.p_x_mean(h1)
        x_mean = x_mean.unsqueeze(2)
        # print('de_x_mean.shape', x_mean.shape) #torch.Size([100, 24, 1])
        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
         对整个模型进行评估，进行编码和解码。注意，log det雅可比矩阵是零
         对于普通VAE(没有流)，并且z_0 = z_k。
        """

        # mean and variance of z
        z_mu, z_var = self.encode(x)

        # sample z
        z = self.reparameterize(z_mu, z_var)
        # print('reparameterize_z',z.shape) # torch.Size([100, 24])

        x_mean = self.decode(z)

        return x_mean, z_mu, z_var, self.log_det_j, z, z


class PlanarVAE(VAE):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, args):
        super(PlanarVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Planar
        self.num_flows = args.num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        self.amor_w = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        # print('encode的input的形状', x.shape) # torch.Size([100, 24, 1])

        batch_size = x.size(0) # 50

        h1,(h2,h3) = self.q_z_nn(x)
        # print('h1.shape',h1.shape) # torch.Size([100, 24, 24])

        # h = h1.view(-1, self.q_z_nn_output_dim)
        h = h1[:, -1, :].unsqueeze(1)#最后一个h
        # print('en_h.shape', h.shape) # torch.Size([100, 1, 24])

        h = h.view(h.size(0), -1)
        # print('en_h2.shape', h.shape)  # torch.Size([100, 24])

        mean_z = self.q_z_mean(h)
        # print('en_mean_z.shape', mean_z.shape) # torch.Size([100, 24])

        var_z = self.q_z_var(h)
        # print('en_var_z.shape', var_z.shape) # torch.Size([100, 24])

        # return amortized u an w for all flows
        u = self.amor_u(h).view(batch_size, self.num_flows, self.z_size, 1)
        # print('u.shape', u.shape) # torch.Size([100, 4, 24, 1])

        w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.z_size)
        # print('w.shape', w.shape) # torch.Size([100, 4, 1, 24])

        b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)
        # print('b.shape', b.shape) # torch.Size([100, 4, 1, 1])

        return mean_z, var_z, u, w, b

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        解码器输出重构图像如下:
        x_mean.shape = (batch_size, num_channels, width, height)
        """
        # print('解码器的输入:z的形状',z.shape) # torch.Size([100, 24])

        z = z.view(z.size(0), self.z_size,1)
        # print('de_z.shape',z.shape)

        h1,(h2,h3) = self.p_x_nn(z)
        # print('de_h1.shape',h1.shape) #de_h1.shape torch.Size([100, 24, 24])

        x_mean = self.p_x_mean(h1)
        # print('de_x_mean.shape', x_mean.shape) #torch.Size([100, 24, 1])
        return x_mean

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var, u, w, b = self.encode(x)
        # print('planer_z_mu',z_mu.shape)
        # print('planer_z_var', z_var.shape)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)] # 100,24

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :], w[:, k, :], b[:, k, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1]) # 100,24
        x_mean = x_mean.squeeze(2) # 删掉第三个维度以便预测值和label的形状相对应
        # print('de_x_mean.shape', x_mean.shape)
        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]
# x 和 x_mean的误差 x_mean和y_test之间还需有个误差

class OrthogonalSylvesterVAE(VAE):
    """
    Variational auto-encoder with orthogonal flows in the encoder.
    """

    def __init__(self, args):
        super(OrthogonalSylvesterVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Sylvester
        self.num_flows = args.num_flows
        self.num_ortho_vecs = args.num_ortho_vecs

        assert (self.num_ortho_vecs <= self.z_size) and (self.num_ortho_vecs > 0)

        # Orthogonalization parameters
        if self.num_ortho_vecs == self.z_size:
            self.cond = 1.e-5
        else:
            self.cond = 1.e-6

        self.steps = 100
        identity = torch.eye(self.num_ortho_vecs, self.num_ortho_vecs)
        # Add batch dimension
        identity = identity.unsqueeze(0)
        # Put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer('_eye', Variable(identity))
        self._eye.requires_grad = False

        # Masks needed for triangular R1 and R2.
        triu_mask = torch.triu(torch.ones(self.num_ortho_vecs, self.num_ortho_vecs), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.num_ortho_vecs).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of R1 * R2 have to satisfy -1 < R1 * R2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs * self.num_ortho_vecs)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs),
            self.diag_activation
        )

        self.amor_q = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.num_ortho_vecs)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.num_ortho_vecs)
            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size * num_flows, z_size * num_ortho_vecs)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, z_size, num_ortho_vecs)
        """

        # Reshape to shape (num_flows * batch_size, z_size * num_ortho_vecs)
        q = q.view(-1, self.z_size * self.num_ortho_vecs)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        amat = torch.div(q, norm)
        dim0 = amat.size(0)
        amat = amat.resize(dim0, self.z_size, self.num_ortho_vecs)

        max_norm = 0.

        # Iterative orthogonalization
        for s in range(self.steps):
            tmp = torch.bmm(amat.transpose(2, 1), amat)
            tmp = self._eye - tmp
            tmp = self._eye + 0.5 * tmp
            amat = torch.bmm(amat, tmp)

            # Testing for convergence
            test = torch.bmm(amat.transpose(2, 1), amat) - self._eye
            norms2 = torch.sum(torch.norm(test, p=2, dim=2) ** 2, dim=1)
            norms = torch.sqrt(norms2)
            max_norm = torch.max(norms).data[0]
            if max_norm <= self.cond:
                break

        if max_norm > self.cond:
            print('\nWARNING WARNING WARNING: orthogonalization not complete')
            print('\t Final max norm =', max_norm)

            print()

        # Reshaping: first dimension is batch_size
        amat = amat.view(-1, self.num_flows, self.z_size, self.num_ortho_vecs)
        amat = amat.transpose(0, 1)

        return amat

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # Amortized r1, r2, q, b for all flows

        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.resize(batch_size, self.num_ortho_vecs, self.num_ortho_vecs, self.num_flows)
        diag1 = diag1.resize(batch_size, self.num_ortho_vecs, self.num_flows)
        diag2 = diag2.resize(batch_size, self.num_ortho_vecs, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        q = self.amor_q(h)
        b = self.amor_b(h)

        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.num_ortho_vecs, self.num_flows)

        return mean_z, var_z, r1, r2, q, b

    def forward(self, x):
        """
        Forward pass with orthogonal sylvester flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var, r1, r2, q, b = self.encode(x)

        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], q_ortho[k, :, :, :], b[:, :, :, k])

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class HouseholderSylvesterVAE(VAE):
    """
    Variational auto-encoder with householder sylvester flows in the encoder.
    """

    def __init__(self, args):
        super(HouseholderSylvesterVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Sylvester
        self.num_flows = args.num_flows
        self.num_householder = args.num_householder
        assert self.num_householder > 0

        identity = torch.eye(self.z_size, self.z_size)
        # Add batch dimension
        identity = identity.unsqueeze(0)
        # Put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer('_eye', Variable(identity))
        self._eye.requires_grad = False

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.z_size, self.z_size), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.z_size).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.z_size)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
            self.diag_activation
        )

        self.amor_q = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.num_householder)

        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.z_size)

            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size, num_flows * z_size * num_householder)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, z_size, z_size)
        """

        # Reshape to shape (num_flows * batch_size * num_householder, z_size)
        q = q.view(-1, self.z_size)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)   # ||v||_2
        v = torch.div(q, norm)  # v / ||v||_2

        # Calculate Householder Matrices
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L

        amat = self._eye - 2 * vvT  # NOTICE: v is already normalized! so there is no need to calculate vvT/vTv

        # Reshaping: first dimension is batch_size * num_flows
        amat = amat.view(-1, self.num_householder, self.z_size, self.z_size)

        tmp = amat[:, 0]
        for k in range(1, self.num_householder):
            tmp = torch.bmm(amat[:, k], tmp)

        amat = tmp.view(-1, self.num_flows, self.z_size, self.z_size)
        amat = amat.transpose(0, 1)

        return amat

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # Amortized r1, r2, q, b for all flows
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.resize(batch_size, self.z_size, self.z_size, self.num_flows)
        diag1 = diag1.resize(batch_size, self.z_size, self.num_flows)
        diag2 = diag2.resize(batch_size, self.z_size, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        q = self.amor_q(h)

        b = self.amor_b(h)

        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.z_size, self.num_flows)

        return mean_z, var_z, r1, r2, q, b

    def forward(self, x):
        """
        Forward pass with orthogonal flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.
        batch_size = x.size(0)

        z_mu, z_var, r1, r2, q, b = self.encode(x)

        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]

            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], q_k, b[:, :, :, k], sum_ldj=True)

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class TriangularSylvesterVAE(VAE):
    """
    Variational auto-encoder with triangular Sylvester flows in the encoder. Alternates between setting
    the orthogonal matrix equal to permutation and identity matrix for each flow.
    """

    def __init__(self, args):
        super(TriangularSylvesterVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.TriangularSylvester
        self.num_flows = args.num_flows

        # permuting indices corresponding to Q=P (permutation matrix) for every other flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.z_size, self.z_size), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.z_size).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.z_size)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
            self.diag_activation
        )

        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.z_size)

            self.add_module('flow_' + str(k), flow_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # Amortized r1, r2, b for all flows
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.resize(batch_size, self.z_size, self.z_size, self.num_flows)
        diag1 = diag1.resize(batch_size, self.z_size, self.num_flows)
        diag2 = diag2.resize(batch_size, self.z_size, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        b = self.amor_b(h)

        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.z_size, self.num_flows)

        return mean_z, var_z, r1, r2, b

    def forward(self, x):
        """
        Forward pass with orthogonal flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var, r1, r2, b = self.encode(x)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            if k % 2 == 1:
                # Alternate with reorderering z for triangular flow
                permute_z = self.flip_idx
            else:
                permute_z = None

            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], b[:, :, :, k], permute_z, sum_ldj=True)

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class IAFVAE(VAE):
    """
    Variational auto-encoder with inverse autoregressive flows in the encoder.
    """

    def __init__(self, args):
        super(IAFVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        self.h_size = args.made_h_size

        self.h_context = nn.Linear(self.q_z_nn_output_dim, self.h_size)

        # Flow parameters
        self.num_flows = args.num_flows
        self.flow = flows.IAF(z_size=self.z_size, num_flows=self.num_flows,
                              num_hidden=24, h_size=self.h_size, conv2d=False)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and context h for flows.
        输出z的基本分布和流的上下文h的参数的编码器。
        """

        h1,(h2,h3) = self.q_z_nn(x)
        h = h1[:, -1, :].unsqueeze(1)#最后一个h
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        h_context = self.h_context(h)

        return mean_z, var_z, h_context

    def forward(self, x):
        """
        Forward pass with inverse autoregressive flows for the transformation z_0 -> z_1 -> ... -> z_k.
        用逆自回归流向前传递转换 z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        对数行列式计算为 log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        # mean and variance of z
        z_mu, z_var, h_context = self.encode(x)
        # sample z
        z_0 = self.reparameterize(z_mu, z_var)

        # iaf flows
        z_k, self.log_det_j = self.flow(z_0, h_context)

        # decode
        x_mean = self.decode(z_k)

        return x_mean, z_mu, z_var, self.log_det_j, z_0, z_k