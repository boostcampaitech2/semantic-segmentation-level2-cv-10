from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RecycleDataset(CustomDataset):
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    CLASSES = ("Background", "General trash", "Paper", "Paper pack", "Metal",
               "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery",
               "Clothing")

    PALETTE = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
               [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
               [192, 192, 128], [64, 64, 128], [128, 0, 192]]

    def __init__(self, **kwargs):
        super(RecycleDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.jpg', **kwargs)
