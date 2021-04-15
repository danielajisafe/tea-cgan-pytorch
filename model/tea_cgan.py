import torch
import torch.nn as nn
import numpy as np

from model.network import Network


class TEACGAN(nn.Module):
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(TEACGAN, self).__init__()
        self._build_modules()

    def _build_modules(self):
        self.image_encoder = Network(self.cfg.network['image_encoder'])
        self.image_decoder = Network(self.cfg.network['image_decoder'])
        self.projection_layer = Network(self.cfg.network['projection_layer'])

        module = [*self.cfg.network['text_encoder'][0].keys()][0]
        args = [*self.cfg.network['text_encoder'][0].values()][0]
        self.text_encoder = eval("nn.{}(**{})".format(module, args))

        self.res_block1 = Network(self.cfg.network['res_block1'])
        self.res_block2 = Network(self.cfg.network['res_block2'])

        self.discriminator = Network(self.cfg.network['discriminator'])
        self.joint_conv = Network(self.cfg.network['joint_conv'])
        self.logit_conv = Network(self.cfg.network['logit_conv'])
        self.uncond_logit_conv = Network(self.cfg.network['logit_conv'])

    def forward(self, image, caption):
        # encode image
        img_ft = self.image_encoder(image)

        # encode text
        N = image.shape[0]
        h_0 = torch.zeros((2, N, 128)).to(self.device)
        c_0 = torch.zeros((2, N, 128)).to(self.device)
        caption_ft, (h_c, c_c) = self.text_encoder(caption, (h_0, c_0))

        # compute attention
        V = torch.flatten(img_ft, start_dim=2)
        W_ = self.projection_layer(caption_ft)
        match_score = torch.bmm(V, W_.permute(0, 2, 1))
        A = nn.Softmax(dim=0)(match_score)
        att_ft = torch.bmm(A, W_)
        att_ft = att_ft.reshape(N, -1, 5, 5)

        # apply residual blocks
        concat_ft = torch.cat([img_ft, att_ft], dim=1)
        res1 = self.res_block1(concat_ft)
        res1 = nn.ReLU()(res1 + concat_ft)

        res2 = self.res_block2(res1)
        res2 = nn.ReLU()(res2 + res1)

        # decode image
        image_hat = self.image_decoder(res2)

        return image_hat, caption_ft

    def discriminator_forward(self, image, caption_ft=None):
        image_ft = self.discriminator(image)

        if caption_ft is not None:
            N = image_ft.shape[0]
            caption_ft = caption_ft.reshape(N, -1, image_ft.shape[-2], image_ft.shape[-1])
            joint_ft = torch.cat([image_ft, caption_ft], dim=1)
            x = self.joint_conv(joint_ft)
        else:
            x = image_ft

        logit = self.logit_conv(x)

        if caption_ft is not None:
            uncond_logit = self.uncond_logit_conv(image_ft)
            return [logit.view(-1), uncond_logit.view(-1)]

        return logit.view(-1)

