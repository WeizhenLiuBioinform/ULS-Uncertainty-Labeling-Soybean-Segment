import os
import shutil
import time
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
import monai
from monai.data import decollate_batch



def train_epoch(
    model: torch.nn.Module, 
    loader: torch.utils.data.Dataloader, 
    optimizer: torch.optim, 
    scaler, 
    epoch: int, 
    loss_func: monai.losses, 
    args: argparse.ArgumentParser,
    ):
    """
    The training results for each epoch.
    
    Args:
    model: segmentation network.
    loader: dataloader
    optimizer: optimizer of segmentation network.
    scaler: Standardized methods.
    epoch: training epoch.
    loss_func: Loss function.
    args: Training parameters configured in main.py.
    
    Returns:
    run_loss.avg: The training loss for each epoch.
    """
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
            #
        if args.amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg

def val_epoch(
    loss_func: monai.losses, 
    model: torch.nn.Module, 
    loader: torch.utils.data.Dataloader, 
    epoch: int, 
    acc_func: function, 
    args: argparse.ArgumentParser, 
    model_inferer=None, 
    post_label=None, 
    post_pred=None, 
    post_sigmoid=None):
    """
    The validation results for each epoch.
    
    Args:
    loss_func: Loss function.
    model: segmentation network.
    loader: dataloader.
    epoch: validation epoch.
    acc_func: the function to calculate IoU.
    args: Training parameters configured in main.py.
    
    Returns:
    run_acc.avg: The validation IoU for each epoch.
    run_loss.avg: The validation loss for each epoch.
    """
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    run_loss = AverageMeter()
    post_sigmoid = post_sigmoid if post_sigmoid is not None else torch.sigmoid
    post_pred = post_pred if post_pred is not None else lambda x: x.argmax(dim=1)

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            loss = loss_func(logits, target)
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_loss.update(loss.item(), n=args.batch_size)
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                print("SSSSSSSSize of run_acc.avg:", len(run_acc.avg))
                if len(run_acc.avg) >= 4:
                    IoU_BG = run_acc.avg[0]
                    IoU_ROD = run_acc.avg[1]
                    IoU_POD = run_acc.avg[2]
                    IoU_SOYBEAN = run_acc.avg[3]
                    
                    print(
                        "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                        ", IoU_BG:", IoU_BG,
                        ", IoU_ROD:", IoU_ROD,
                        ", IoU_POD:", IoU_POD,
                        ", IoU_SOYBEAN:", IoU_SOYBEAN,
                        ",val_loss",run_loss.avg,
                        ", time {:.2f}s".format(time.time() - start_time),
                    )
                else:
                    avg_acc = np.mean(run_acc.avg)
                    print(
                        "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                        "loss",run_loss.avg,
                        "acc",
                        avg_acc,
                        "time {:.2f}s".format(time.time() - start_time),
                    )
            start_time = time.time()
    return run_acc.avg,run_loss.avg



def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model: torch.nn.Module,
    train_loader: torch.utils.data.Dataloader,
    val_loader: torch.utils.data.Dataloader,
    optimizer: torch.optim,
    loss_func: monai.losses,
    acc_func: function,
    args: argparse.ArgumentParser,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    """
    Train segmentation network.
    
    Args:
    model: segmentation network.
    train_loader: train dataloader.
    val_loader: val dataloader.
    optimizer: optimizer of segmentation network.
    loss_func: Loss function.
    acc_func: Function to calculate IoU.
    args: Training parameters configured in main.py.
    
    Returns:
    val_acc_max: The max value of the IoU on the validation set across all training epochs.
    """
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc,val_loss = val_epoch(
                loss_func,
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            if args.rank == 0:
                IoU_BG, IoU_ROD, IoU_POD, IoU_SOYBEAN= val_acc
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", IoU_BG:",
                    IoU_BG,
                    ", IoU_ROD:",
                    IoU_ROD,
                    ", IoU_POD:",
                    IoU_POD,
                    ", IoU_SOYBEAN:",
                    IoU_SOYBEAN,
                    ", val_loss:",val_loss,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_IoU_bg", IoU_BG, epoch)
                    writer.add_scalar("val_IoU_rod", IoU_ROD, epoch)
                    writer.add_scalar("val_IoU_pod", IoU_POD, epoch)
                    writer.add_scalar("val_IoU_soybean", IoU_SOYBEAN, epoch)
                    writer.add_scalar("val_loss", val_loss, epoch)
            val_avg_acc = np.mean(val_acc)

            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                b_new_best = True
                if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(
                        model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                    )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
