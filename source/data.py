import os, sys
from libs import *

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_dir, 
        image_size = 512, 
        augment = False, 
    ):
        self.image_files, self.mask_files = sorted(glob.glob(data_dir + "images/*")), sorted(glob.glob(data_dir + "masks/*"))
        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(), A.ShiftScaleRotate(rotate_limit = 15), 
                    A.ColorJitter(), 
                    A.CLAHE(), 
                    A.Resize(
                        height = image_size, width = image_size, 
                    ), 
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(
                        height = image_size, width = image_size, 
                    ), 
                ]
            )

    def __len__(self, 
    ):
        return len(self.image_files)

    def __getitem__(self, 
        index, 
    ):
        image_file, mask_file = self.image_files[index], self.mask_files[index]
        image, mask = np.load(image_file), np.load(mask_file)
        if self.transform is not None:
            T = self.transform(image = image, mask = mask)
            image, mask = T["image"], T["mask"]

        image = A.Normalize()(image = image)["image"]

        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)