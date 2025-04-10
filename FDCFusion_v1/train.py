import argparse
import time
import random
import sys
from loss import Fusionloss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from models import TCM
import os


torch.backends.cudnn.deterministic=True     #保证每次运行的结果一致，这对于需要可重复实验结果的科学研究非常重要。
torch.backends.cudnn.benchmark=False        #保证每次运行的速度一致，自动优化可能会导致每次运行的结果略有不同。
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    #指定使用哪一块GPU，如使用第二块GPU，则填写"1"。



# 配置优化器-------------------------------------------------------------
def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
         lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


#训练一个epoch--------------------------------------------------------
def train_one_epoch(model, criterion, train_dataloader1,train_dataloader2, optimizer, aux_optimizer, epoch, clip_max_norm):

    model.train()
    device = next(model.parameters()).device

    for i, d1, d2 in zip(range(len(train_dataloader1)), train_dataloader1, train_dataloader2):

        d1 = d1.to(device)
        d2 = d2.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d1,d2)

        out_criterion = criterion(out_net, d1, d2)
        out_criterion["loss"].backward()
        optimizer.step()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 1 == 0:
            print(
                f"{time.ctime()}"
                f"\tTrain epoch {epoch}: ["
                f"{i * len(d1)}/{len(train_dataloader1.dataset)}"
                f" ({100. * i / len(train_dataloader1):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tloss_in: {out_criterion["loss_in"].item():.3f} |'
                f'\tloss_grad: {out_criterion["loss_grad"].item():.3f} |'
                f'\tloss_fre_total: {out_criterion["loss_fre_total"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )



#保存模型-----------------------------------------------------------
def save_checkpoint(state, epoch, save_path, filename):
    torch.save(state, save_path + "checkpoint.pth.tar")


#配置参数---------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-d", "--dataset", type=str, default=r"C:\Users\ZYY\Desktop\CFNet\CFNet-master\input\MSRS-main\train", help="Training dataset"
    )
    parser.add_argument('--A_dir', type=str, default='vi',
                        help='input test image name')
    parser.add_argument('--B_dir', type=str, default='ir',
                        help='input test image name')
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    # parser.add_argument("--checkpoint", type=str, default="D:\Xml\compression\LIC_TCM0829maskjoint\save_path/1checkpoint_latest.pth.tar", help="Path to a checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--save_path", type=str, default='./save_path', help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    args = parser.parse_args(argv)
    return args


#主程序-------------------------------------------------------------
def main(argv):
    #获取参数字典
    args = parse_args(argv)
    #打印参数名和键值
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    #生成路径
    save_path = os.path.join(args.save_path, str(args.lmbda))
    A_dir = os.path.join(args.dataset, args.A_dir)
    B_dir = os.path.join(args.dataset, args.B_dir)

    #创建保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #设置随机种子，以确保实验的可重复性
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    #首先对图像进行随机裁剪，然后将裁剪后的图像转换为张量。
    train_transforms1 = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()])

    train_transforms2 = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor(), transforms.Grayscale()]
    )


    A_dataset = ImageFolder(A_dir, split="train", transform=train_transforms1)
    B_dataset = ImageFolder(B_dir, split="train", transform=train_transforms2)


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda'


    train_dataloader1 = DataLoader(
        A_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    train_dataloader2 = DataLoader(
        B_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    #加载网络
    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320, isRGB=True)
    net = net.to(device)

    #优化器
    optimizer, aux_optimizer = configure_optimizers(net, args)


    milestones = args.lr_epoch
    print("milestones: ", milestones)

    #通过调整学习率可以加速模型的收敛，并提高模型的性能。
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    #损失函数
    criterion = Fusionloss()

    last_epoch = 0
    #可以在中断训练后继续训练，不用从头开始
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


    #开始训练
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader1,
            train_dataloader2,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        #学习率调度器允许你在训练过程中根据一定的策略动态调整学习率。
        lr_scheduler.step()

        #保存模型
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )



if __name__ == "__main__":
    main(sys.argv[1:])
