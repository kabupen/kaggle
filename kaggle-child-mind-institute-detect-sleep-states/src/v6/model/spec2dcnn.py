from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

from augmentation.cutmix import Cutmix
from augmentation.mixup import Mixup


class Spec2DCNN(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder_name: str,
        in_channels: int,
        encoder_weights: Optional[str] = None,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
        is_submit = False
    ):
        super().__init__()

        """
        feature_extractor : 
            - input : [bs, #特徴量, #時系列] 
            - output : [bs, C, H, #時系列] 
            - 表形式の #特徴量 x #時系列 のデータを U-Net へ入力するために画像情報に変換する

        encoder : 
            - input :  [bs, C, H, #時系列] 
            - output :  [bs, C, H, #時系列] 
            - このときに H (height) と #時系列 (width) は 32 の倍数である必要がある
            >>> RuntimeError: Wrong input shape height=128, width=1021. Expected image height and width divisible by 32. Consider pad your images to shape (128, 1024).
        
        decoder:
            - input :  [bs, C*H, #時系列] 
            - output :  [bs, #時系列, #y_pred] 
        """

        # feat extractor
        self.feature_extractor = feature_extractor

        # encoder
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None if is_submit else encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        
        # decoder
        self.decoder = decoder

        # augmentation
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)

    def forward(
        self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        
        # 特徴量抽出
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)

        # augmentation
        if do_mixup and y_true is not None:
            x, y_true = self.mixup(x, y_true)
        if do_cutmix and y_true is not None:
            x, y_true = self.cutmix(x, y_true)

        # encoder
        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)

        # decoder
        x = self.decoder(x)  # (batch_size, n_timesteps, n_classes)

        return x