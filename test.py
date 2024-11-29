import os.path
from collections import OrderedDict

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image


import PuffNet
import transformer
import extraction
from argument import args
from thop import profile


args.train = False

# Device
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

vgg = PuffNet.vgg
vgg.load_state_dict(torch.load("vgg_normalised.pth"))
vgg = nn.Sequential(*list(vgg.children())[:44])

base = extraction.BaseFeatureExtraction()
detail = extraction.DetailFeatureExtraction()
decoder = PuffNet.decoder
embedding = PuffNet.PatchEmbed()
Trans = transformer.Transformer()
decoder.eval()
embedding.eval()
Trans.eval()

# model load
state_dict = torch.load('base_iter_12000.pth')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
base.load_state_dict(new_state_dict)

state_dict = torch.load('decoder_iter_100000.pth')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

state_dict = torch.load('detail_iter_100000.pth')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
detail.load_state_dict(new_state_dict)

state_dict = torch.load('embedding_iter_100000.pth')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

state_dict = torch.load('transformer_iter_100000.pth')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)


network = PuffNet.Puff(vgg, decoder, embedding, base, detail, Trans, args)
network.eval()


network.to(device)

# ToTensor
def test_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

tf = test_transform()

# You can try our model by using it
content_images = Image.open('content/content.jpg')
style_images = Image.open('style/style.jpg')

content_images = tf(content_images)
style_images = tf(style_images)

content_images = content_images.unsqueeze(0)
style_images = style_images.unsqueeze(0)

if torch.cuda.is_available():
    content_images = content_images.cuda()
    style_images = style_images.cuda()




with torch.no_grad():
    out = network(content_images, style_images)
if os.path.exists('res/test'):
    os.mkdir('res/test')
output_name = '{:s}/{:s}{:s}'.format(
            "res/test", "test", ".jpg"
)

out = out.to(torch.device('cpu'))
save_image(out, output_name)




