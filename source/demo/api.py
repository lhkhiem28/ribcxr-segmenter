import os, sys
from libs import *

class Seg():
    def __init__(self, 
        ckp_dir, 
    ):
        self.transform = A.Compose(
            [
                A.Normalize(), AT.ToTensorV2(), 
            ]
        )

        self.model = torch.load(
            ckp_dir, 
            map_location = "cpu", 
        ).eval()
        self.assigned_colors = {
            "R1" :np.random.rand(3), 
            "R2" :np.random.rand(3), 
            "R3" :np.random.rand(3), 
            "R4" :np.random.rand(3), 
            "R5" :np.random.rand(3), 
            "R6" :np.random.rand(3), 
            "R7" :np.random.rand(3), 
            "R8" :np.random.rand(3), 
            "R9" :np.random.rand(3), 
            "R10":np.random.rand(3), 
            "L1" :np.random.rand(3), 
            "L2" :np.random.rand(3), 
            "L3" :np.random.rand(3), 
            "L4" :np.random.rand(3), 
            "L5" :np.random.rand(3), 
            "L6" :np.random.rand(3), 
            "L7" :np.random.rand(3), 
            "L8" :np.random.rand(3), 
            "L9" :np.random.rand(3), 
            "L10":np.random.rand(3), 
        }

    def seg_predict(self, 
        image_file, 
    ):
        image = Image.open(image_file).convert("RGB")
        image = np.asarray(image, dtype = np.uint8)
        image = A.Resize(
            height = 512, width = 512, 
        )(image = image)["image"]
        pred = self.model(self.transform(image = image)["image"].unsqueeze(0)) > 0.5
        pred = pred.int().squeeze(0).permute(1, 2, 0).numpy()

        from detectron2.utils.visualizer import Visualizer, ColorMode
        visualizer = Visualizer(
            image, instance_mode = ColorMode.SEGMENTATION, 
        )
        output = visualizer.overlay_instances(
            masks = pred.transpose(2, 0, 1), 
            labels = list(self.assigned_colors.keys()), assigned_colors = list(self.assigned_colors.values()), 
        ).get_image()

        return output