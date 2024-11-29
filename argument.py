import argparse


# Training Argument
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--cs_weight', type=float, default=20.0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--content_data', type=str, default='data/train2014')
parser.add_argument('--style_data', type=str, default='data/wikiart')
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--train', type=bool, default=True)
args = parser.parse_args()

