from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.tof_dataset import *
from geoseg.models.DCSwin import dcswin_small
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 100
ignore_index = 255
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 1e-3
backbone_lr = 1e-5
backbone_weight_decay = 1e-3
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "dcswin"
weights_path = "model_weights/tof/{}".format(weights_name)
test_weights_name = "dcswin"
log_name = "tof/{}".format(weights_name)
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None  # the path for the pretrained model weight
gpus = "auto"  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = dcswin_small(num_classes=num_classes)

# define the loss
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)

use_aux_loss = False

train_dataset = TOFDataset(
    data_root="data/tof/train", mode="test", transform=train_aug  # mosaic_ratio=0.25,
)

val_dataset = TOFDataset(data_root="data/tof/val", transform=val_aug, mode="test")
test_dataset = TOFDataset(
    data_root="data/tof/test",
    transform=val_aug,
    mode="test",
    img_dir="images_5000",
    mask_dir="masks_5000",
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

# define the optimizer
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
