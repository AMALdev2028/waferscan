"""
Networks. Every backbone - the small CPU-trainable WaferNet or a big timm
model (EfficientNet-B7, SE-ResNeXt-101, Swin) - ends in the same ClassHead:

    feature maps F (B,K,h,w) -> global average pool g (B,K) -> dropout ->
        fc   : class logits      (softmax head, trained with focal loss)
        evid : evidence logits   (evidential head, Dirichlet uncertainty)

Why this shape pays off at inference:
- CAM heatmap for free: with GAP followed by a linear layer, the class
  activation map is just sum_k W[c,k] * F[k] - no gradients needed, so it
  also works from an exported ONNX/TensorRT graph.
- MC dropout for free: dropout sits only on g, so T stochastic predictions
  are T tiny matrix products on g (numpy), not T full network passes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018): learn *which
    channels* matter (channel attention) and *where* on the wafer to look
    (spatial attention)."""

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(ch, ch // reduction, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(ch // reduction, ch, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


def _block(cin: int, cout: int, pool: bool) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              CBAM(cout)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ClassHead(nn.Module):
    def __init__(self, k: int, num_classes: int, drop: float):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(k, num_classes)
        self.evid = nn.Linear(k, num_classes)

    def forward(self, f: torch.Tensor):
        g = f.mean((2, 3))
        d = self.drop(g)
        return self.fc(d), self.evid(d), g, f


class WaferNet(nn.Module):
    """~0.6M-parameter CNN for 2x64x64 die-map inputs. Trains on a laptop CPU;
    also the edge-fallback model for offline fab-floor stations."""

    def __init__(self, num_classes: int = 9, in_ch: int = 2, widths=(24, 48, 96, 160), drop: float = 0.3):
        super().__init__()
        chans = [in_ch, *widths]
        self.features = nn.Sequential(*[_block(chans[i], chans[i + 1], pool=i < len(widths) - 1)
                                        for i in range(len(widths))])
        self.num_features = widths[-1]
        self.head = ClassHead(self.num_features, num_classes, drop)

    def forward(self, x):
        return self.head(self.features(x))


class TimmNet(nn.Module):
    """Any timm backbone + our ClassHead. The die map is upsampled to the
    backbone's native input size (e.g. 224) inside forward()."""

    def __init__(self, name: str, num_classes: int = 9, in_ch: int = 2, drop: float = 0.3,
                 pretrained: bool = False, input_size: int = 224):
        super().__init__()
        import timm
        self.backbone = timm.create_model(name, pretrained=pretrained, in_chans=in_ch, num_classes=0)
        self.num_features = self.backbone.num_features
        self.input_size = input_size
        self.head = ClassHead(self.num_features, num_classes, drop)

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        f = self.backbone.forward_features(x)
        if f.ndim == 4 and f.shape[-1] == self.num_features and f.shape[1] != self.num_features:
            f = f.permute(0, 3, 1, 2)                      # Swin returns NHWC
        return self.head(f)


# name -> (constructor kwargs). The three big ones are the comparison asked
# for; "wafernet" is what this repo trains on CPU and ships by default.
BACKBONES = {
    "wafernet": {},
    "efficientnet_b7": {"timm": "tf_efficientnet_b7.ns_jft_in1k", "input_size": 224},  # plain efficientnet_b7 has no pretrained weights in timm
    "seresnext101": {"timm": "seresnext101_32x8d", "input_size": 224},     # ResNeXt-101 + squeeze-excitation attention
    "swin_t": {"timm": "swin_tiny_patch4_window7_224", "input_size": 224},  # ViT with hierarchical patch merging
}


def build_model(name: str = "wafernet", num_classes: int = 9, drop: float = 0.3, pretrained: bool = False) -> nn.Module:
    spec = BACKBONES[name]
    if name == "wafernet":
        return WaferNet(num_classes=num_classes, drop=drop)
    return TimmNet(spec["timm"], num_classes=num_classes, drop=drop, pretrained=pretrained,
                   input_size=spec["input_size"])


class Ensemble(nn.Module):
    """K members in one graph -> one ONNX/TensorRT engine for the deep
    ensemble. Outputs are stacked on dim 1: (B, K, ...)."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        outs = [m(x) for m in self.members]
        return tuple(torch.stack([o[i] for o in outs], 1) for i in range(4))
