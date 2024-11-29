import os.path

import torch
from torch import nn
from torchvision.utils import save_image
from collections import OrderedDict

import PuffNet
import transformer
import extraction
from argument import args
from sample import content_iter, style_iter




# Device
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# vgg
vgg = PuffNet.vgg
vgg.load_state_dict(torch.load("vgg_normalised.pth"))
vgg = nn.Sequential(*list(vgg.children())[:44])

# decoder
decoder = PuffNet.decoder
embedding = PuffNet.PatchEmbed()

# feature extraction
base = extraction.BaseFeatureExtraction()
detail = extraction.DetailFeatureExtraction()
Trans = transformer.Transformer()


# base load
# state_dict = torch.load('base_iter_10000.pth')
# new_state_dict = OrderedDict()
# for k,v in state_dict.items():
#     namekey=k
#     new_state_dict[namekey]=v
# base.load_state_dict(new_state_dict)
#

with torch.no_grad():
    network = PuffNet.Puff(vgg, decoder, embedding, base, detail, Trans, args)

network.train()

network.to(device)

optimizer = torch.optim.Adam([
    {'params': network.transformer.parameters()},
    {'params': network.decode.parameters()},
    {'params': network.embedding.parameters()},
    {'params': network.detail.parameters()},
    {'params': network.base.parameters()},
], lr=args.lr)


loss_sum = []

for i in range(1000000):

    if i < 400000:
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    out, c, s, loss_fe, loss_c, loss_s, l_identity1, l_identity2 = network(content_images, style_images)

    if (i+1) % 10 == 0:
        print("train_epoch:{:d}".format(i+1))
    if (i+1) % 100 == 0:
        if not os.path.exists('res'):
            os.mkdir('res')
        if not os.path.exists('res/res'):
            os.mkdir('res/res')
        # result image
        output_name = '{:s}/{:s}{:s}'.format(
             "res/res", str(i), ".jpg"
        )

        out = torch.cat((content_images, out), 0)
        out = torch.cat((style_images, out), 0)
        out = out.to(torch.device('cpu'))
        save_image(out, output_name)

        # feature image
        if not os.path.exists('res/content'):
            os.mkdir('res/content')
        if not os.path.exists('res/style'):
            os.mkdir('res/style')
        c = torch.cat((content_images, c), 0)
        s = torch.cat((style_images, s), 0)
        save_image(c, 'res/content/c{:d}.jpg'.format(i+1))
        save_image(s, 'res/style/s{:d}.jpg'.format(i+1))

    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_fe = args.cs_weight * loss_fe
    loss = loss_fe + loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)

    # loss = loss.requires_grad_(True)

    if (i + 1) % 1000 == 0:
        loss_sum.append(loss)
        print("train_epoch:{:d},loss:{:f}".format(i + 1, loss))
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()
    torch.cuda.empty_cache()
    if (i + 1) % 5000 == 0 and (i + 1) >= 5000:
        if not os.path.exists('model'):
            os.mkdir('model')
        state_dict = network.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format("model", i + 1))

        state_dict = network.decode.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format("model", i + 1))

        state_dict = network.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/embedding_iter_{:d}.pth'.format("model", i + 1))

        state_dict = network.detail.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/detail_iter_{:d}.pth'.format("model", i + 1))

        state_dict = network.base.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/base_iter_{:d}.pth'.format("model", i + 1))

print(loss_sum)


