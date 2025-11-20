import torch
import torch.nn as nn
import torch.nn.functional as F


class Unik3DDepthModule(nn.Module):
    def __init__(self, cfg, model_name="lpiccinelli/unik3d-vitb"):
        super().__init__()
        self.cfg = cfg

        print(f"Loading Unik3D model: {model_name}")
        try:
            from unik3d.models import UniK3D
            self.unik3d = UniK3D.from_pretrained(model_name)
            self.unik3d.eval()
            for param in self.unik3d.parameters():
                param.requires_grad = False
            print("✓ Unik3D loaded successfully (frozen)")
        except ImportError:
            raise ImportError(
                "Unik3D not installed. Install from: https://github.com/lpiccinelli-eth/UniK3D"
            )

        self.d_bound = cfg.d_bound
        self.depth_bins = torch.arange(
            self.d_bound[0],
            self.d_bound[1],
            self.d_bound[2]
        )
        self.num_bins = len(self.depth_bins)

        self.temperature = nn.Parameter(
            torch.tensor(getattr(cfg, 'unik3d_temperature', 1.0)),
            requires_grad=False
        )

        self.use_rays = getattr(cfg, 'use_unik3d_rays', False)

    def metric_to_distribution(self, metric_depth):
        B, H, W = metric_depth.shape
        D = self.num_bins
        device = metric_depth.device

        depth_bins = self.depth_bins.to(device)

        metric_depth = metric_depth.unsqueeze(1)
        depth_bins = depth_bins.view(1, D, 1, 1)

        distances = torch.abs(metric_depth - depth_bins)
        weights = torch.exp(-distances / self.temperature)
        depth_prob = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        return depth_prob

    def forward(self, images):
        B, C, H, W = images.shape

        if images.max() <= 1.0:
            images = images * 255.0

        outputs = []
        metric_depths = []
        rays_list = [] if self.use_rays else None

        with torch.no_grad():
            for i in range(B):
                rgb = images[i]
                predictions = self.unik3d.infer(rgb)

                metric_depth = predictions["depth"]
                metric_depths.append(metric_depth)

                depth_prob = self.metric_to_distribution(
                    metric_depth.unsqueeze(0)
                )
                outputs.append(depth_prob)

                if self.use_rays:
                    rays = predictions["rays"]
                    rays_list.append(rays)

        depth_prob = torch.cat(outputs, dim=0)
        metric_depth = torch.stack(metric_depths, dim=0)

        if self.use_rays:
            rays = torch.stack(rays_list, dim=0)
            return depth_prob, metric_depth, rays
        else:
            return depth_prob, metric_depth, None
