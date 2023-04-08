import os, sys
from libs import *

class Seg():
    def __init__(self, 
        ckp_dir, 
    ):
        self.model = torch.load(
            ckp_dir, 
            map_location = "cpu", 
        )
        self.transform = A.Compose(
            [
                A.Resize(
                    height = 512, width = 512, 
                ), 
                A.Normalize(), AT.ToTensorV2(), 
            ]
        )

    def seg_predict(self, 
        image_file, 
    ):
        image = Image.open(image_file).convert("RGB")
        image = np.asarray(image, dtype = np.uint8)

        pred = self.model(self.transform(image = image)["image"].unsqueeze(0)) > 0.75
        pred = pred.int().squeeze(0).permute(1, 2, 0).numpy()

        return pred