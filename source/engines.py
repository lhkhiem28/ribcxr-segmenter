import os, sys
from libs import *

def train_fn(
    train_loaders, num_epochs, 
    model, 
    optimizer, 
    device = torch.device("cpu"), 
    save_ckp_dir = "./", 
):
    print("\nStart Training ...\n" + " = "*16)
    model = model.to(device)

    best_metric = 0.0
    for epoch in range(1, num_epochs + 1):
        print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)

        model.train()
        running_loss, running_metric,  = 0.0, 0.0, 
        for images, masks in tqdm.tqdm(train_loaders["train"]):
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss, metric,  = catalyst.contrib.losses.dice.DiceLoss()(logits, masks), catalyst.metrics.functional.dice(
                logits, masks
                , threshold = 0.5, mode = "macro"
            ), 

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss, running_metric,  = running_loss + loss.item()*images.size(0), running_metric + metric.item()*images.size(0), 
        train_loss, train_metric,  = running_loss/len(train_loaders["train"].dataset), running_metric/len(train_loaders["train"].dataset), 
        wandb.log({"train_loss":train_loss, "train_metric":train_metric, }, step = epoch)
        print("{:<8} - loss:{:.4f}, metric:{:.4f}".format(
            "train", 
            train_loss, train_metric, 
        ))

        with torch.no_grad():
            model.eval()
            running_loss, running_metric,  = 0.0, 0.0, 
            for images, masks in tqdm.tqdm(train_loaders["val"]):
                images, masks = images.to(device), masks.to(device)

                logits = model(images)
                loss, metric,  = catalyst.contrib.losses.dice.DiceLoss()(logits, masks), catalyst.metrics.functional.dice(
                    logits, masks
                    , threshold = 0.5, mode = "macro"
                ), 

                running_loss, running_metric,  = running_loss + loss.item()*images.size(0), running_metric + metric.item()*images.size(0), 
        val_loss, val_metric,  = running_loss/len(train_loaders["val"].dataset), running_metric/len(train_loaders["val"].dataset), 
        wandb.log({"val_loss":val_loss, "val_metric":val_metric, }, step = epoch)
        print("{:<8} - loss:{:.4f}, metric:{:.4f}".format(
            "val", 
            val_loss, val_metric, 
        ))
        if val_metric > best_metric:
            torch.save(model, "{}/best.ptl".format(save_ckp_dir))
            best_metric = val_metric

    print("\nFinish Training ...\n" + " = "*16)
    return {
        "train_loss":train_loss, "train_metric":train_metric, 
        "val_loss":val_loss, "val_metric":val_metric, 
    }

def test_fn(
    test_loader, 
    model, 
    device = torch.device("cpu"), 
):
    print("\nStart Testing ...\n" + " = "*16)
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        running_loss, running_metric,  = 0.0, 0.0, 
        for images, masks in tqdm.tqdm(test_loader):
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss, metric,  = catalyst.contrib.losses.dice.DiceLoss()(logits, masks), catalyst.metrics.functional.dice(
                logits, masks
                , threshold = 0.5, mode = "macro"
            ), 

            running_loss, running_metric,  = running_loss + loss.item()*images.size(0), running_metric + metric.item()*images.size(0), 
    test_loss, test_metric,  = running_loss/len(test_loader.dataset), running_metric/len(test_loader.dataset), 
    print("{:<8} - loss:{:.4f}, metric:{:.4f}".format(
        "test", 
        test_loss, test_metric, 
    ))

    print("\nFinish Testing ...\n" + " = "*16)
    return {
        "test_loss":test_loss, "test_metric":test_metric, 
    }