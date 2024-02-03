
from model.cnnspectrogram import CNNSpectrogram
from model.unet1d import UNet1DDecoder
from model.spec2dcnn import Spec2DCNN
from model.spec1d import Spec1D

def get_model(cfg):
    if cfg.model.name=="Spec2DCNN":
        feat_ext = CNNSpectrogram(
            in_channels=len(cfg.feature_names),
            output_size=cfg.n_frames,
            **cfg.feature_extractor.CNNSpectrogram,
        )
        decoder = UNet1DDecoder(
            n_channels=feat_ext.height,
            n_classes=cfg.n_classes,
            duration=cfg.n_frames,
            **cfg.decoder.UNet1DDecoder,
        )
        model = Spec2DCNN(
            feature_extractor=feat_ext,
            decoder=decoder,
            in_channels=feat_ext.out_chans,
            is_submit=cfg.submit.flag,
            **cfg.model.Spec2DCNN,
        )
    elif cfg.model.name=="Spec1D":
        feat_ext = CNNSpectrogram(
            in_channels=len(cfg.feature_names),
            output_size=cfg.n_frames,
            **cfg.feature_extractor.CNNSpectrogram,
        )
        decoder = UNet1DDecoder(
            n_channels=feat_ext.height,
            n_classes=cfg.n_classes,
            duration=cfg.n_frames,
            **cfg.decoder.UNet1DDecoder,
        )
        model = Spec1D(
            feature_extractor=feat_ext,
            decoder=decoder,
            in_channels=feat_ext.out_chans,
            is_submit=cfg.submit.flag,
            **cfg.model.Spec2DCNN,
        )
    return model