#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhaoyang Sun

import networks
from networks import init_net
import torch
import torch.nn as nn
import torch.nn.functional as F


class MakeupGAN(nn.Module):
    def __init__(self, opts):
        super(MakeupGAN, self).__init__()
        self.opts = opts

        # parameters
        self.lr = opts.lr
        self.batch_size = opts.batch_size

        self.gpu = torch.device('cuda:{}'.format(opts.gpu)) if opts.gpu >= 0 else torch.device('cpu')
        self.input_dim = opts.input_dim
        self.output_dim = opts.output_dim
        self.semantic_dim = opts.semantic_dim

        # encoders
        self.enc_content = init_net(networks.E_content(opts.input_dim), opts.gpu, init_type='normal', gain=0.02)
        self.enc_makeup = init_net(networks.E_makeup(opts.input_dim), opts.gpu, init_type='normal', gain=0.02)
        self.enc_semantic = init_net(networks.E_semantic(opts.semantic_dim), opts.gpu, init_type='normal', gain=0.02)
        self.transformer = init_net(networks.Transformer(), opts.gpu, init_type='normal', gain=0.02)
        # generator
        self.gen = init_net(networks.Decoder(opts.output_dim), opts.gpu, init_type='normal', gain=0.02)

    def forward(self, non_makeup, makeup, non_makeup_parse, makeup_parse):
        # first transfer and removal
        z_non_makeup_c = self.enc_content(non_makeup)
        z_non_makeup_s = self.enc_semantic(non_makeup_parse)
        z_non_makeup_a = self.enc_makeup(non_makeup)

        z_makeup_c = self.enc_content(makeup)
        z_makeup_s = self.enc_semantic(makeup_parse)
        z_makeup_a = self.enc_makeup(makeup)
        # warp makeup style
        mapX, mapY, z_non_makeup_a_warp, z_makeup_a_warp = self.transformer(z_non_makeup_c,
                                                                            z_makeup_c,
                                                                            z_non_makeup_s,
                                                                            z_makeup_s,
                                                                            z_non_makeup_a,
                                                                            z_makeup_a)
        # makeup transfer and removal
        z_transfer = self.gen(z_non_makeup_c, z_makeup_a_warp)
        z_removal = self.gen(z_makeup_c, z_non_makeup_a_warp)

        # rec
        z_rec_non_makeup = self.gen(z_non_makeup_c, z_non_makeup_a)
        z_rec_makeup = self.gen(z_makeup_c, z_makeup_a)

        # second transfer and removal
        z_transfer_c = self.enc_content(z_transfer)
        # z_non_makeup_s = self.enc_semantic(non_makeup_parse)
        z_transfer_a = self.enc_makeup(z_transfer)

        z_removal_c = self.enc_content(z_removal)
        # z_makeup_s = self.enc_semantic(makeup_parse)
        z_removal_a = self.enc_makeup(z_removal)
        # warp makeup style
        mapX2, mapY2, z_transfer_a_warp, z_removal_a_warp = self.transformer(z_transfer_c,
                                                                             z_removal_c,
                                                                             z_non_makeup_s,
                                                                             z_makeup_s,
                                                                             z_transfer_a,
                                                                             z_removal_a)
        # makeup transfer and removal
        z_cycle_non_makeup = self.gen(z_transfer_c, z_removal_a_warp)
        z_cycle_makeup = self.gen(z_removal_c, z_transfer_a_warp)

        return z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        self.enc_content.load_state_dict(checkpoint['enc_c'])
        self.enc_makeup.load_state_dict(checkpoint['enc_a'])
        self.enc_semantic.load_state_dict(checkpoint['enc_s'])
        self.transformer.load_state_dict(checkpoint['enc_trans'])
        self.gen.load_state_dict(checkpoint['gen'])
        return checkpoint['ep'], checkpoint['total_it']

    def normalize_image(self, x):
        return x[:, 0:3, :, :]

    def test_pair(self, data):
        non_makeup = data['non_makeup'].to(self.gpu).detach()
        makeup = data['makeup'].to(self.gpu).detach()
        non_makeup_parse = data['non_makeup_parse'].to(self.gpu).detach()
        makeup_parse = data['makeup_parse'].to(self.gpu).detach()
        with torch.no_grad():
            # first transfer and removal
            z_non_makeup_c = self.enc_content(non_makeup)
            z_non_makeup_s = self.enc_semantic(non_makeup_parse)
            z_non_makeup_a = self.enc_makeup(non_makeup)

            z_makeup_c = self.enc_content(makeup)
            z_makeup_s = self.enc_semantic(makeup_parse)
            z_makeup_a = self.enc_makeup(makeup)
            # warp makeup style
            mapX, mapY, z_non_makeup_a_warp, z_makeup_a_warp = self.transformer(z_non_makeup_c,
                                                                                z_makeup_c,
                                                                                z_non_makeup_s,
                                                                                z_makeup_s,
                                                                                z_non_makeup_a,
                                                                                z_makeup_a)
            # makeup transfer and removal
            z_transfer = self.gen(z_non_makeup_c, z_makeup_a_warp)
            z_removal = self.gen(z_makeup_c, z_non_makeup_a_warp)

        non_makeup_down = self.normalize_image(F.interpolate(non_makeup, scale_factor=0.25, mode='nearest'))
        n, c, h, w = non_makeup_down.shape
        non_makeup_down_warp = torch.bmm(non_makeup_down.view(n, c, h * w), self.mapY)  # n*HW*1
        non_makeup_down_warp = non_makeup_down_warp.view(n, c, h, w)
        non_makeup_warp = F.interpolate(non_makeup_down_warp, scale_factor=4)

        makeup_down = self.normalize_image(F.interpolate(makeup, scale_factor=0.25, mode='nearest'))
        n, c, h, w = makeup_down.shape
        makeup_down_warp = torch.bmm(makeup_down.view(n, c, h * w), self.mapX)  # n*HW*1
        makeup_down_warp = makeup_down_warp.view(n, c, h, w)
        makeup_warp = F.interpolate(makeup_down_warp, scale_factor=4)

        images_non_makeup = self.normalize_image(non_makeup).detach()
        images_makeup = self.normalize_image(makeup).detach()
        images_z_transfer = self.normalize_image(z_transfer).detach()
        row1 = torch.cat((images_non_makeup[0:1, ::], images_makeup[0:1, ::], makeup_warp[0:1, ::], images_z_transfer[0:1, ::]), 3)
        return row1


class NonMakeupD(nn.Module):
    def __init__(self, opts):
        super(NonMakeupD, self).__init__()
        if opts.dis_scale > 1:
            self.dis = networks.MultiScaleDis(opts.input_dim, n_scale=opts.dis_scale, norm=opts.dis_norm)
        else:
            self.dis = networks.Dis(opts.input_dim, norm=opts.dis_norm)
        return

    def forward(self, x):
        return self.dis(x)


class MakeupD(nn.Module):
    def __init__(self, opts):
        super(MakeupD, self).__init__()
        if opts.dis_scale > 1:
            self.dis = networks.MultiScaleDis(opts.input_dim, n_scale=opts.dis_scale, norm=opts.dis_norm)
        else:
            self.dis = networks.Dis(opts.input_dim, norm=opts.dis_norm)

    def construct(self, x):
        return self.dis(x)
