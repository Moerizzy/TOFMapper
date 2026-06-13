"""RGB + NIR + nDSM (5 bands), trained on the shared 5-band stack.
Bands 1, 2, 3, 4, 5 = R, G, B, NIR, nDSM."""

from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.tof_dataset_multiband_norm import CLASSES
from geoseg.models.FTUNetFormer_multiBand import ft_unetformer
from tools.multimodal_datasets import build_datasets
from tools.oversampling import make_weighted_sampler
from tools.utils import Lookahead, process_model_params

# ----- experiment ----------------------------------------------------------
DATA_ROOT = "data/tof"
BAND_INDICES = (1, 2, 3, 4, 5)  # R, G, B, NIR, nDSM
IN_CHANS = len(BAND_INDICES)
RUN_TAG = "rgbn_ndsm"

# Class-aware oversampling on the train set. Disable for a vanilla baseline.
USE_OVERSAMPLING = False
OVERSAMPLE_CLASSES = (2, 3, 4)  # Patch, Linear, Tree (rare TOF classes)
OVERSAMPLE_METHOD = "inverse_freq"  # or "presence"

# ----- training hparams ----------------------------------------------------
max_epoch = 100
ignore_index = 255
train_batch_size = 4
val_batch_size = 4
accumulate_grad_batches = 2  # effektive Batch-Size = 8
precision = "bf16-mixed"
lr = 6e-4
weight_decay = 1e-3
backbone_lr = 1e-5
backbone_weight_decay = 1e-3
num_classes = len(CLASSES)
classes = CLASSES

weights_name = f"ftunetformer_{RUN_TAG}_all_regions"
weights_path = f"TOFMapper/model_weights/ortho_all_{RUN_TAG}/{weights_name}"
test_weights_name = weights_name
log_name = f"ortho_all_{RUN_TAG}/{weights_name}"
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = [1]
resume_ckpt_path = None

# ----- model ---------------------------------------------------------------
net = ft_unetformer(
    pretrained=True,
    num_classes=num_classes,
    in_chans=IN_CHANS,
    weight_path="pretrain_weights/stseg_base.pth",
)

# ----- loss ----------------------------------------------------------------
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False

# ----- data ----------------------------------------------------------------
train_dataset, val_dataset, test_dataset = build_datasets(
    data_root=DATA_ROOT, band_indices=BAND_INDICES
)

if USE_OVERSAMPLING:
    train_sampler = make_weighted_sampler(
        data_root=DATA_ROOT,
        oversample_classes=OVERSAMPLE_CLASSES,
        method=OVERSAMPLE_METHOD,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
else:
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

# ----- optimizer -----------------------------------------------------------
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
