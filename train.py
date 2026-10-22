import os
import time
import datetime

import torch
from torch.utils import data

from src.u2net import u2net_full, u2net_lite
from src.unet import UNet
from train_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
from my_dataset import KidneyDataset
import transforms as T


class SegmentationPresetTrain:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    if train:
        return SegmentationPresetTrain(mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(U2Net: bool = True, num_classes: int = 1):
    if U2Net:
        model = u2net_lite(num_classes)
    else:
        model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    # 由compute_mean_std.py计算得出
    mean = (0.343, 0.343, 0.343)
    std = (0.186, 0.186, 0.186)
    # 用来保存训练以及验证过程中信息
    if args.u2net is True:
        results_file = "u2net_results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        results_file = "unet_results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_dataset = KidneyDataset(args.data_path,
                                  train=True,
                                  transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset = KidneyDataset(args.data_path,
                                train=False,
                                transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = 2
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        collate_fn=train_dataset.collate_fn)

    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      collate_fn=val_dataset.collate_fn)

    model = create_model(U2Net=args.u2net, num_classes=num_classes)
    model.to(device)

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 训练代码
    best_dice = 0.
    start_time = time.time()
    if args.u2net:
        print("U2Net start training")
    else:
        print("UNet start training")
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch, num_classes=num_classes,
                                        lr_scheduler=lr_scheduler, U2Net=args.u2net, print_freq=args.print_freq, scaler=scaler)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        confmat, dice = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"Val Dice: {dice:.3f}")

        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"

            val_info = f"Val_Dice: {dice:.3f}\n" \
                       f"{val_info}\n"

            f.write(train_info + val_info + "\n")

        if best_dice < dice:
            best_dice = dice
            if args.u2net:
                torch.save(save_file, "./save_weights/u2net_best_model.pth")
            else:
                torch.save(save_file, "./save_weights/unet_best_model.pth")
        else:
            continue

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net training")

    parser.add_argument("--data-path", default="./", help="Kidney root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=10, type=int, help="validation interval default 10 Epochs")

    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--u2net", action="store_true", default=True)
    parser.add_argument("--unet", dest="u2net", action="store_false")
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
