import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, input, reference):
        temp_a = torch.sum(F.normalize(input, dim=2) * F.normalize(reference, dim=2), 2, keepdim=True)
        temp_b = torch.sum(F.normalize(input, dim=3) * F.normalize(reference, dim=3), 3, keepdim=True)
        a = torch.sum(temp_a, keepdim=False)
        b = torch.sum(temp_b, keepdim=False)
        B, c, h, w = input.shape
        return -(a + b) / h


class GPLoss(nn.Module):
    def __init__(self):
        super(GPLoss, self).__init__()
        self.trace = SPLoss()

    def get_image_gradients(self, input):
        f_v_1 = input[:, :, :, :-1]
        f_v_2 = input[:, :, :, 1:]
        f_v = f_v_1 - f_v_2

        f_h_1 = input[:, :, :-1, :]
        f_h_2 = input[:, :, 1:, :]
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def forward(self, input, reference):
        # comment these lines when you inputs and outputs are in [0,1] range already
        input = (input + 1) / 2
        reference = (reference + 1) / 2

        input_v, input_h = self.get_image_gradients(input)
        ref_v, ref_h = self.get_image_gradients(reference)

        trace_v = self.trace(input_v, ref_v)
        trace_h = self.trace(input_h, ref_h)
        return trace_v + trace_h


class CPLoss(nn.Module):
    def __init__(self, rgb=True, yuv=True, yuvgrad=True):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.trace = SPLoss()
        self.trace_YUV = SPLoss()

    def get_image_gradients(self, input):
        f_v_1 = input[:, :, :, :-1]
        f_v_2 = input[:, :, :, 1:]
        f_v = f_v_1 - f_v_2

        f_h_1 = input[:, :, :-1, :]
        f_h_2 = input[:, :, 1:, :]
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def to_YUV(self, input):
        return torch.concat((0.299 * torch.unsqueeze(input[:, 0, :, :], 1) +
                             0.587 * torch.unsqueeze(input[:, 1, :, :], 1) +
                             0.114 * torch.unsqueeze(input[:, 2, :, :], 1), \
                             0.493 * (torch.unsqueeze(input[:, 2, :, :], 1) - (
                                     0.299 * torch.unsqueeze(input[:, 0, :, :], 1) +
                                     0.587 * torch.unsqueeze(input[:, 1, :, :], 1) +
                                     0.114 * torch.unsqueeze(input[:, 2, :, :], 1))), \
                             0.877 * (torch.unsqueeze(input[:, 0, :, :], 1) - (
                                     0.299 * torch.unsqueeze(input[:, 0, :, :], 1) +
                                     0.587 * torch.unsqueeze(input[:, 1, :, :], 1) +
                                     0.114 * torch.unsqueeze(input[:, 2, :, :], 1)))), dim=1)

    def forward(self, input, reference):
        # comment these lines when you inputs and outputs are in [0,1] range already
        input = (input + 1) / 2
        reference = (reference + 1) / 2
        total_loss = 0
        if self.rgb:
            total_loss += self.trace(input, reference)
        if self.yuv:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            total_loss += self.trace(input_yuv, reference_yuv)
        if self.yuvgrad:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            input_v, input_h = self.get_image_gradients(input_yuv)
            ref_v, ref_h = self.get_image_gradients(reference_yuv)

            total_loss += self.trace(input_v, ref_v)
            total_loss += self.trace(input_h, ref_h)

        return total_loss
