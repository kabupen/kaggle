
from model.cnnspectrogram import CNNSpectrogram
from model.unet1d import UNet1DDecoder
from model.spec2dcnn import Spec2DCNN

def get_model(cfg):
    feat_ext = CNNSpectrogram(
        in_channels=len(cfg.feature_names),
        output_size=cfg.n_frames,
        **cfg.feature_extractor.CNNSpectrogram,
        # base_filters=cfg.base_filters,
        # kernel_sizes=cfg.kernel_sizes,
        # stride=cfg.stride,
        # in_channels=len(cfg.feature_names),
        # output_size=cfg.n_frames,
        # sigmoid=True,
        # reinit=True
    )
    decoder = UNet1DDecoder(
        n_channels=feat_ext.height,
        n_classes=cfg.n_classes,
        duration=cfg.n_frames,
        **cfg.decoder.UNet1DDecoder,
        
        # n_channels=feat_ext.height,
        # n_classes=cfg.n_classes,
        # duration=cfg.n_frames,
        # bilinear=cfg.bilinear,
        # se=cfg.se,
        # res=cfg.res,
        # scale_factor=cfg.scale_factor,
        # dropout=cfg.dropout,
    )
    model = Spec2DCNN(
        feature_extractor=feat_ext,
        decoder=decoder,
        in_channels=feat_ext.out_chans,
        is_submit=cfg.submit.flag,
        **cfg.model.Spec2DCNN,
        # encoder_name=cfg.encoder_name,
        # encoder_weights=cfg.encoder_weights,
        # mixup_alpha=1,
        # cutmix_alpha=1
    )
    return model