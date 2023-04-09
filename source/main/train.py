import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from engines import *

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/RibCXR-Seg/train/", 
            augment = True, 
        ), 
        num_workers = 12, batch_size = 16, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/RibCXR-Seg/val/", 
            augment = False, 
        ), 
        num_workers = 12, batch_size = 16, 
        shuffle = False, 
    ), 
}
model = smp.Unet(
    encoder_name = "efficientnet-b1", encoder_weights = "imagenet", 
    classes = 20, activation = "sigmoid", 
)
optimizer = torch.optim.Adam(
    model.parameters(), lr = 1e-3, 
)

wandb.init(
    entity = "khiemlhfx", project = "RibCXR-Seg", 
    name = "efficientunet", 
)
save_ckp_dir = "../../ckps/RibCXR-Seg/efficientunet"
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, num_epochs = 200, 
    model = model, 
    optimizer = optimizer, 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    save_ckp_dir = save_ckp_dir, 
)