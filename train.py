import itertools
import torch
import os
import os.path as osp
import torch.nn.functional as F
from torchvision.utils import save_image
from model import MakeupGAN, MakeupD, NonMakeupD
import torch.nn.init as init
from losses import GANLoss, GPLoss, CPLoss
from dataset_makeup import MakeupDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from options import MakeupOptions


class SSAT:
    def __init__(self, opts):
        self.G = MakeupGAN(opts)
        self.D_nonmakup = NonMakeupD(opts)
        self.D_makup = MakeupD(opts)

        self.adv_loss = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.criterionL1 = torch.nn.L1Loss()
        self.GPL = GPLoss()
        self.CPL = CPLoss(rgb=True, yuv=True, yuvgrad=True)

        self.CP_weight = opts.CP_weight
        self.GP_weight = opts.GP_weight
        self.rec_weight = opts.rec_weight
        self.cycle_weight = opts.cycle_weight
        self.semantic_weight = opts.semantic_weight
        self.adv_weight = opts.adv_weight

        self.G.apply(self.weights_init_xavier)
        self.D_nonmakup.apply(self.weights_init_xavier)
        self.D_makup.apply(self.weights_init_xavier)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), opts.lr, (opts.beta1, opts.beta2))

        parameters = [self.D_nonmakup.parameters()]
        parameters.insert(0, self.D_makup.parameters())
        self.d_optimizer = torch.optim.Adam(itertools.chain(*parameters), opts.lr, (opts.beta1, opts.beta2))

        self.opts = opts
        return

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if isinstance(m, torch.nn.Conv2d):
            init.xavier_normal(m.weight.data, gain=1.0)
        elif isinstance(m, torch.nn.Linear):
            init.xavier_normal(m.weight.data, gain=1.0)

    def load_checkpoint(self, snapshot_path):
        G_path = os.path.join(snapshot_path, 'G.pth')
        if os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path))
            print('loaded trained generator {}..!'.format(G_path))
        D_makup_path = os.path.join(snapshot_path, 'D_makup.pth')
        if os.path.exists(D_makup_path):
            self.D_nonmakup.load_state_dict(torch.load(D_makup_path))
            print('loaded trained discriminator D_makup {}..!'.format(D_makup_path))

        D_nonmakup_path = os.path.join(snapshot_path, 'D_nonmakup.pth')
        if os.path.exists(D_nonmakup_path):
            self.D_nonmakup.load_state_dict(torch.load(D_nonmakup_path))
            print('loaded trained discriminator D_nonmakup {}..!'.format(D_nonmakup_path))
        return

    def train(self):

        dataset = MakeupDataset(self.opts)
        # """
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.opts.batch_size,
                                shuffle=True, num_workers=self.opts.nThreads)

        start = self.opts.start_epoch

        for e in range(start, self.opts.max_epoch):
            is_save = False
            for i, data in enumerate(dataloader):
                non_makeup = data['non_makeup']
                makeup = data['makeup']
                transfer_g = data['transfer']
                removal_g = data['removal']
                non_makeup_parse = data['non_makeup_parse']
                makeup_parse = data['makeup_parse']

                # ============= Train G ========================#

                z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY = self.G(
                    non_makeup, makeup, non_makeup_parse, makeup_parse)

                # Ladv for generator
                loss_G_GAN_non_makeup = self.adv_loss(self.D_nonmakup(z_removal), True)
                loss_G_GAN_makeup = self.adv_loss(self.D_makup(z_transfer), True)
                loss_G_GAN = (loss_G_GAN_non_makeup + loss_G_GAN_makeup) * 0.5 * self.adv_weight

                # rec loss
                loss_G_rec_non_makeup = self.criterionL1(non_makeup, z_rec_non_makeup)
                loss_G_rec_makeup = self.criterionL1(makeup, z_rec_makeup)
                loss_G_rec = (loss_G_rec_non_makeup + loss_G_rec_makeup) * 0.5 * self.rec_weight

                # cycle loss
                loss_G_cycle_non_makeup = self.criterionL1(non_makeup, z_cycle_non_makeup)
                loss_G_cycle_makeup = self.criterionL1(makeup, z_cycle_makeup)
                loss_G_cycle = (loss_G_cycle_non_makeup + loss_G_cycle_makeup) * 0.5 * self.cycle_weight

                # semantic loss
                non_makeup_parse_down = F.interpolate(non_makeup_parse, size=(self.opts.crop_size // 4, self.opts.crop_size // 4), mode='nearest')
                n, c, h, w = non_makeup_parse_down.shape
                non_makeup_parse_down_warp = torch.bmm(non_makeup_parse_down.reshape(n, c, h * w), mapY)  # n*HW*1
                non_makeup_parse_down_warp = non_makeup_parse_down_warp.reshape(n, c, h, w)

                makeup_parse_down = F.interpolate(makeup_parse, size=(self.opts.crop_size // 4, self.opts.crop_size // 4), mode='nearest')
                n, c, h, w = makeup_parse_down.shape
                makeup_parse_down_warp = torch.bmm(makeup_parse_down.reshape(n, c, h * w), mapX)  # n*HW*1
                makeup_parse_down_warp = makeup_parse_down_warp.reshape(n, c, h, w)

                loss_G_semantic_non_makeup = self.criterionL1(non_makeup_parse_down, makeup_parse_down_warp)
                loss_G_semantic_makeup = self.criterionL1(makeup_parse_down, non_makeup_parse_down_warp)
                loss_G_semantic = (loss_G_semantic_makeup + loss_G_semantic_non_makeup) * 0.5 * self.semantic_weight

                # makeup loss
                loss_G_CP = self.CPL.construct(z_transfer, transfer_g) + self.CPL.construct(z_removal, removal_g)
                loss_G_GP = self.GPL.construct(z_transfer, non_makeup) + self.GPL.construct(z_removal, makeup)
                loss_G_SPL = loss_G_CP * self.CP_weight + loss_G_GP * self.GP_weight

                loss_G = loss_G_GAN + loss_G_rec + loss_G_cycle + loss_G_semantic + loss_G_SPL

                self.g_optimizer.zero_grad()
                loss_G.backward(retain_graph=False)
                self.g_optimizer.step()

                # =============================== Train D ============================= #
                non_makeup_real = self.D_nonmakup(non_makeup)
                non_makeup_fake = self.D_nonmakup(Variable(z_removal.data).detach())
                makeup_real = self.D_makup(makeup)
                makeup_fake = self.D_makup(Variable(z_transfer.data).detach())
                loss_D_non_makeup = self.adv_loss(non_makeup_fake, False) + self.adv_loss(non_makeup_real, False)
                loss_D_makeup = self.adv_loss(makeup_fake, False) + self.adv_loss(makeup_real, False)
                loss_D = (loss_D_makeup + loss_D_non_makeup) * 0.5
                self.d_optimizer.zero_grad()
                loss_D.backward(retain_graph=False)
                self.d_optimizer.step()

                # Decay learning rate
                if (e + 1) % self.opts.snapshot_step == 0 and not is_save:
                    self.save_models(e, i, non_makeup, makeup, mapX, mapY, z_transfer, z_removal, transfer_g, removal_g,
                                     z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup)

    def save_models(self, e, i, non_makeup, makeup, mapX, mapY, z_transfer, z_removal, transfer_g, removal_g,
                    z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup):

        if not osp.exists(self.opts.snapshot_path):
            os.makedirs(self.opts.snapshot_path)
        torch.save(
            self.G.state_dict(),
            os.path.join(
                self.opts.snapshot_path, '{}_{}_G.pth'.format(e + 1, i + 1)))
        torch.save(
            self.D_nonmakup.state_dict(),
            os.path.join(
                self.opts.snapshot_path, '{}_{}_D_nonmakup.pth'.format(e + 1, i + 1)))
        torch.save(
            self.D_makup.state_dict(),
            os.path.join(
                self.opts.snapshot_path, '{}_{}_D_makup.pth'.format(e + 1, i + 1)))

        _, C, H, W = non_makeup.shape

        non_makeup_down = F.interpolate(non_makeup, size=(H // 4, W // 4))
        n, c, h, w = non_makeup_down.shape
        non_makeup_down_warp = torch.bmm(non_makeup_down.reshape(n, c, h * w), mapY)  # n*HW*1
        non_makeup_down_warp = non_makeup_down_warp.reshape(n, c, h, w)
        non_makeup_warp = F.interpolate(non_makeup_down_warp, size=(H, W))

        makeup_down = F.interpolate(makeup, size=(H // 4, W // 4))
        n, c, h, w = makeup_down.shape
        makeup_down_warp = torch.bmm(makeup_down.reshape(n, c, h * w), mapX)  # n*HW*1
        makeup_down_warp = makeup_down_warp.reshape(n, c, h, w)
        makeup_warp = F.interpolate(makeup_down_warp, size=(H, W))

        row_1 = torch.cat((non_makeup, makeup_warp, transfer_g, z_transfer, z_rec_non_makeup, z_cycle_non_makeup), dim=3)
        row_2 = torch.cat((makeup, non_makeup_warp, removal_g, z_removal, z_rec_makeup, z_cycle_makeup), dim=3)
        result = torch.cat((row_1, row_2), dim=2)
        save_image(result, os.path.join(self.opts.imgs_dir, f"{e}_result.jpg"), normalize=True)


if __name__ == '__main__':
    parser = MakeupOptions()
    opts = parser.parse()
    ssat = SSAT(opts)
    ssat.train()