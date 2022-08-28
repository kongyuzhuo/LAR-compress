import os
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import LRHR_PKLDataset
from res_vq import res_vqvae
from torchvision import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--cause_sr_resume', type=str, default='/GPFS/data/bsguo/LAR-all/backbone_srresnet/0.0023_model_epoch_24.pth')
    parser.add_argument("--GT_size", type=int, default=160)
    parser.add_argument("--train_dataroot_GT", type=str, default='pkls/DIV2K_train_HR.pklv4')
    parser.add_argument("--train_dataroot_LQ", type=str, default='pkls/DIV2K_train_HR_X4.pklv4')
    parser.add_argument("--val_dataroot_GT", type=str, default='pkls/DIV2K_valid_HR.pklv4')
    parser.add_argument("--val_dataroot_LQ", type=str, default='pkls/DIV2K_valid_HR_X4.pklv4')
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--stage', type=str, default='train')

    args = parser.parse_args()
    return args


def main(args):
    dataset = LRHR_PKLDataset(args.GT_size, args.train_dataroot_GT, args.train_dataroot_LQ)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    model = res_vqvae(stage=args.stage, cause_sr_resume=args.cause_sr_resume, backbone='srresnet').cuda()

    if args.resume:
        print('===>load resume ', args.resume)
        ckpt = torch.load(args.resume)
        # ckpt = {key:value for (key, value) in ckpt.items() if '_t.' in key or 'stage_one' in key}
        print(ckpt.keys())
        model.load_state_dict(ckpt, strict=True)
    # model.cause_sr.load_state_dict(torch.load(args.cause_sr_resume)["model"].state_dict())
    # model = nn.DataParallel(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for i in range(args.epoch):
        train(i, loader, model, optimizer)
        torch.save(model.state_dict(), f"checkpoint_resvq_resnet/vqvae_{str(i + 1).zfill(3)}.pt")


def train(epoch, loader, model, optimizer):
    loader = tqdm(loader)
    criterion = nn.MSELoss()
    model.train()
    model.cause_sr.eval()

    latent_loss_weight = 0.25
    sample_size = 10
    recon_loss_sum = 0.
    latent_loss_sum = 0.
    cause_recon_loss_sum = 0.

    for i, data in enumerate(loader):
        HR = data['GT'].cuda().float()
        LR = data['LQ'].cuda().float()

        model.zero_grad()


        out, latent_loss, cause_sr = model(HR, LR)

        cause_recon_loss = criterion(cause_sr, HR)
        recon_loss = criterion(out, HR)

        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        recon_loss_sum += recon_loss.detach().cpu().data
        latent_loss_sum += latent_loss.detach().cpu().data
        cause_recon_loss_sum += cause_recon_loss.detach().cpu().data
        loader.set_description(
            (
                f"epoch: {epoch + 1}; mse: {recon_loss_sum / (i + 1):.5f}; cause mse: {cause_recon_loss_sum / (i + 1):.5f}"
                f"latent: {latent_loss_sum / (i + 1):.3f}; "
                f"lr: {lr:.5f}"
            )
        )
        if i % 100 == 0:
            model.eval()

            sample = HR[:sample_size]
            sample_lr = LR[:sample_size]
            with torch.no_grad():
                out, latent_loss, cause_sr = model(sample, sample_lr)

                utils.save_image(
                    (torch.cat([F.upsample(sample_lr, scale_factor=4, mode='bicubic'), sample, out, cause_sr], dim=-1)),
                    f"sample_resvq_resnet/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=1,
                    # normalize=True,
                    # range=(-1, 1),
                )

            model.train()


if __name__ == '__main__':
    args = get_args()
    main(args)
