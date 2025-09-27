import argparse
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from torch.nn.functional import l1_loss
from module.adaface_load import load_adaface_backbone, encode_with_adaface
from module.perceptual_loss import PerceptualLoss
try:
    import wandb

except ImportError:
    wandb = None
from torch.optim.lr_scheduler import LambdaLR
from model import Generator, Discriminator
from module.dataset import MultiResolutionDataset
from module.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def one_hot(dim, tmp):
    ones = torch.eye(dim)
    return ones.index_select(0, tmp)


def make_label(dim, batch):
    tmp = torch.LongTensor(np.random.randint(dim, size=batch))
    code = one_hot(dim, tmp)
    label = torch.LongTensor(tmp)
    return code, label

def make_specific_label(label, dim, batch):
    tmp = torch.LongTensor([label] * batch)
    code = one_hot(dim, tmp)
    return code

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def train(args, train_loader, val_loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    train_loader = sample_data(train_loader)
    val_loader = sample_data(val_loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0 
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator
        
    accum = 0.5 ** (32 / (10 * 1000))
    z_repeat = args.n_sample // args.batch
    sample_z = []
    for _ in range(z_repeat):
        sample, _ = next(val_loader)
        sample_z.append(sample)
    sample_z = torch.stack(sample_z).view(args.n_sample, 3, args.size, args.size)
    utils.save_image(
                     sample_z,
                     f'{sample_path}/real.png',
                     nrow=int(args.n_sample ** 0.5),
                     normalize=True,
                     value_range=(-1, 1),
                    )
    sample_z = sample_z.to(device)
    
    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print('Done!')
            break
    
        train_img, train_atr = next(train_loader)
        train_atr_one = torch.from_numpy(np.eye(10)[train_atr]).type(torch.FloatTensor)
        
        train_img = train_img.to(device)
        train_atr = train_atr.to(device)
        train_atr_one = train_atr_one.to(device)

        f_code, f_label = make_label(10, args.batch) # one-hot encoding
        f_code, f_label = f_code.to(device), f_label.to(device)


##########################trainD##########################

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        with torch.no_grad():
            fake_img, _, train_feat = generator(train_img, f_code)
            rec_img,  _, fake_feat = generator(fake_img, train_atr_one)
        fake_pred,_ = discriminator(fake_img, f_label)
        rec_pred,_ = discriminator(rec_img, train_atr)

        real_pred,real_cls = discriminator(train_img, train_atr)

        c_loss = F.cross_entropy(real_cls,train_atr) 
        d_loss = d_logistic_loss(real_pred, fake_pred)+d_logistic_loss(real_pred, rec_pred)+c_loss*1.2
        loss_dict['d'] = d_loss 
        loss_dict['c'] = c_loss 
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step() 
        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            train_img.requires_grad = True
            real_pred,_ = discriminator(train_img, train_atr)
            r1_loss = d_r1_loss(real_pred, train_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss
##########################trainG###########################
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        fake_img, _,train_feat = generator(train_img, f_code) 
        rec_img, _,fake_feat = generator(fake_img,train_atr_one)
        _, _ ,rec_feat= generator(rec_img, train_atr_one)

        fake_pred,fake_cls = discriminator(fake_img, f_label)
        rec_pred,rec_cls = discriminator(rec_img,train_atr)
        train_pred,train_cls = discriminator(train_img,train_atr)

        fc_fake_loss = F.cross_entropy(fake_cls, f_label)
        fc_rec_loss = F.cross_entropy(rec_cls, f_label)
        fc_train_loss = F.cross_entropy(train_cls,train_atr)
        fc_loss = fc_fake_loss * 0.5 + fc_train_loss + fc_rec_loss *0.5
        
        rec_loss = l1_loss(train_img,rec_img)

        normed_real = encode_with_adaface(backbone, train_img.to(device))  # [B, D]
        normed_fake = encode_with_adaface(backbone, fake_img.to(device))  # [B, D]

        # 4. similarity loss
        csim = F.cosine_similarity(normed_real, normed_fake, dim=1)  # [B]
        id_loss = 1.0 - csim.mean()  # want fake close to real
        perc_loss = perceptual_loss_fn(train_img, fake_img)

        w_fc  = 1.0
        w_id  = 10.0
        w_rec = 8.0 
        w_perc = 5.0 
        #######################
        g_loss = (
            g_nonsaturating_loss(fake_pred)
        + g_nonsaturating_loss(rec_pred)
        + fc_loss * w_fc
        + id_loss * w_id
        + rec_loss * w_rec
        + perc_loss * w_perc
        )

        #######################

        loss_dict['g']    = g_loss
        loss_dict['id']   = id_loss
        loss_dict['fc']   = fc_loss
        loss_dict['rec']  = rec_loss
        loss_dict['perc'] = perc_loss
        
        g_loss.backward()
        g_optim.step()
        g_scheduler.step()

        accumulate(g_ema, g_module, accum)
        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        id_val = loss_reduced['id'].mean().item()
        rec_val = loss_reduced['rec'].mean().item()
        c_val = loss_reduced['c'].mean().item()
        fc_val = loss_reduced['fc'].mean().item()
        perc_loss_val = loss_reduced['perc'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; c: {c_val}'
                    f'rec: {rec_val:.4f}; c: {c_val:.4f}; f_c: {fc_val:.4f}; id: {id_val:.4f}; perc: {perc_loss_val:.4f}; '
                )
            )
                # trip: {trip_val:.4f}; 

            if wandb and args.wandb:
                wandb.log(
                    {
                        'Generator': g_loss_val,
                        'Discriminator': d_loss_val,
                        'R1': r1_val,
                        'Mean Path Length': mean_path_length,
						'ID': id_val,
                        'rec': rec_val,
                        'classification loss': c_val,
                        'La': fc_val,
 			            'Real Class': c_val,
                        'Perceptual Loss': perc_loss_val,
                        'lr': g_optim.param_groups[0]['lr'],
                    }
                )

            if i % 1000 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    old_code = make_specific_label(9, 10, sample_z.size(0)).to(device)
                    sample, _,_ = g_ema(sample_z, old_code)
                    utils.save_image(
                        sample,
                        f'{sample_path}//{str(i).zfill(6)}_old.png',
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    yg_code = make_specific_label(0, 10, sample_z.size(0)).to(device)
                    sample, _,_ = g_ema(sample_z, yg_code)
                    utils.save_image(
                        sample,
                        f'{sample_path}//{str(i).zfill(6)}_young.png',
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )

                    sample,_,_ = g_ema(fake_img, train_atr_one)
                    utils.save_image(
                        sample,
                        f'{sample_path}/{str(i).zfill(6)}_rec.png',
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )

            if i % 10000 == 0:
                main_ckpt = {
                    'g': g_module.state_dict(),
                    'd': d_module.state_dict(),
                    'g_ema': g_ema.state_dict(),
                    'g_optim': g_optim.state_dict(),
                    'd_optim': d_optim.state_dict(),
                    }
                main_path = os.path.join(checkpoint_path, f'{i:06d}.pt')
                torch.save(main_ckpt, main_path)

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default="./data/train_128_balance")
    parser.add_argument('--val_path', type=str, default="./data/val_128_relabel_ori")
    parser.add_argument('--iter', type=int, default=600000)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--n_sample', type=int, default=8)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--r1', type=float, default=200)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--wandb_project', type=str, default='Agetransformer_github')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("-n", "--name", type=str, required=True, help="experiment name")
    parser.add_argument('--resume', action='store_true', default=False, help="from last checkpoint")
    parser.add_argument('--ckpt', type=str, default=None)

    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1
    perceptual_loss_fn = PerceptualLoss(layers=('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')).to(device)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    backbone, device = load_adaface_backbone("ir_50", "./models/adaface_ir50_ms1mv2.ckpt")
    
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr* g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    sample_path = f'sample/{args.name}'
    checkpoint_path = f'checkpoint/{args.name}'
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        generator.load_state_dict(ckpt['g'],     strict=False)
        discriminator.load_state_dict(ckpt['d'], strict=False)
        g_ema.load_state_dict(ckpt['g_ema'],     strict=False)
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
        args.start_iter = int(os.path.splitext(os.path.basename(args.ckpt))[0])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    for pg in g_optim.param_groups:
        if 'initial_lr' not in pg:
            pg['initial_lr'] = pg['lr']
    g_scheduler = LambdaLR(g_optim, lr_lambda=lambda step: 1.0, last_epoch=args.start_iter)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.train_path, transform, args.size)
    train_loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )


    dataset = MultiResolutionDataset(args.val_path, transform, args.size)
    val_loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.login(key="YOUR_WANDB_KEY")
        wandb.init(project=args.wandb_project, name=args.name)
    train(args, train_loader, val_loader, generator, discriminator, g_optim, d_optim, g_ema, device)
