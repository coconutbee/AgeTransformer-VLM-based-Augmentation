import argparse

import torch
from torchvision import utils
import torchvision
# from model_spade import Generator
from model_ori import Generator
from dataset import MultiResolutionDataset
from torchvision import transforms, utils
from torch.utils import data
from train import one_hot
import os
import os
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_extensions'


def generate(args, loader, g_ema, device):
    count = 0
    with torch.no_grad():
        g_ema.eval()
        for batch_data in loader:
            img = batch_data[0].to(device)
            label = batch_data[1]
            if args.age==5:
                label[label == args.age] = 5
            elif args.age!=0:
                label[label == args.age] = 0
            else:
                label[label == args.age] = 1
            label[label == 5] = args.age
            label = one_hot(10, label).to(device)

            minibatch_size = len(img)
            sample, _,_ = g_ema(img, label)
            for j in range(minibatch_size):
                age = label[j].argmax().item()
                if age == args.age:
                    if args.age == 0:
                        utils.save_image(
                img[j],
                args.save_path+'/real/{}.png'.format(str(count).zfill(6)),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
                    utils.save_image(
            sample[j],
            args.save_path+'/fake{}/{}.png'.format(args.age, str(count).zfill(6)),
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
                count += 1

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--age', type=int, default=8)
    parser.add_argument('--mode', type=str, default="progression")
    parser.add_argument('--save_path', type=str, default="./agetransformer_output")
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="./models/160000.pt")
    parser.add_argument('--val_path', type=str, default="./data/caf_enhanced")
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint['g_ema'])


    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.val_path, transform, args.size)
    val_loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        drop_last=False,
    )
    for i in range(0,10):
        args.age = i
        if not os.path.exists(args.save_path+'/real'):
            os.makedirs(args.save_path+'/real')
        if not os.path.exists(args.save_path+'/fake{}'.format(args.age)):
            os.makedirs(args.save_path+'/fake{}'.format(args.age))
        generate(args, val_loader, g_ema, device)

