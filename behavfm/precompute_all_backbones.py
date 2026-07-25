"""Precompute patch features for all 5 backbones on AP-10K + APT-36K.

Extracts frozen backbone features from 5 vision foundation models and
caches as .npy files. This is the one-time compute cost that makes
all downstream training cheap (architecture.md section 1.4).

Backbones (all ViT-B/16 or equivalent, 768-dim output):
  - DINOv2-base   (facebook/dinov2-base)          14x14 patches -> 256 tokens
  - DINOv1-base   (facebook/dino-vitb16)           16x16 patches -> 196 tokens
  - MAE-base      (facebook/vit-mae-base)          16x16 patches -> 196 tokens
  - CLIP-ViT-B/16 (openai/clip-vit-base-patch16)  16x16 patches -> 196 tokens
  - EVA-02-B/16   (timm: eva02_base_patch16_clip_224) 16x16 patches -> 196 tokens

Cache layout per backbone:
    out_dir/{backbone_name}/
      features.npy          float16 (N, n_patches, 768)
      features.shape.json   shape + dtype metadata
      meta.json             per-item metadata records

Usage:
    python precompute_all_backbones.py \
        --ap10k-root data/ap10k/ap-10k \
        --out-dir data/cache/multi_backbone \
        --backbones dinov2,dino,mae,clip,eva02 \
        --split-file ap10k-train-split1.json \
        --species-filter all \
        --device cuda

    # APT-36K (requires separate data root):
    python precompute_all_backbones.py \
        --apt36k-root data/apt36k \
        --out-dir data/cache/multi_backbone_apt36k \
        --backbones dinov2,clip \
        --device cuda
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
# Allow importing from the BehaviorFM source tree if available.
_SRC_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent.parent / "src",
    Path(os.environ.get("BEHAVFM_SRC", "src")),
]
for _src in _SRC_CANDIDATES:
    if (_src / "iti").is_dir():
        sys.path.insert(0, str(_src.parent if _src.name == "src" else _src))
        sys.path.insert(0, str(_src))
        break

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompute_all_backbones")


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY = {
    "dinov2": {
        "hf_id": "facebook/dinov2-base",
        "loader": "hf_dinov2",
        "patch_size": 14,
        "hidden_dim": 768,
        "description": "DINOv2-base (self-distillation)",
    },
    "dino": {
        "hf_id": "facebook/dino-vitb16",
        "loader": "hf_dino",
        "patch_size": 16,
        "hidden_dim": 768,
        "description": "DINOv1-base (self-distillation)",
    },
    "mae": {
        "hf_id": "facebook/vit-mae-base",
        "loader": "hf_mae",
        "patch_size": 16,
        "hidden_dim": 768,
        "description": "MAE-base (masked reconstruction)",
    },
    "clip": {
        "hf_id": "openai/clip-vit-base-patch16",
        "loader": "hf_clip",
        "patch_size": 16,
        "hidden_dim": 768,
        "description": "CLIP-ViT-B/16 (language-image contrastive)",
    },
    "eva02": {
        "hf_id": "timm/eva02_base_patch16_clip_224.merged2b",
        "loader": "timm_eva02",
        "patch_size": 16,
        "hidden_dim": 768,
        "description": "EVA-02-B/16 (MIM on CLIP features)",
    },
}


class FrozenBackbone(nn.Module):
    """Unified frozen backbone wrapper for all 5 architectures.

    All backbones are frozen (eval mode, no grad). Output is always
    patch tokens only (CLS dropped), shape (B, n_patches, 768).
    """

    def __init__(self, backbone_name: str, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        info = BACKBONE_REGISTRY[backbone_name]
        self.hf_id = info["hf_id"]
        self.patch_size = info["patch_size"]
        self.hidden_dim = info["hidden_dim"]
        self.loader_type = info["loader"]

        if self.loader_type == "hf_dinov2":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.hf_id, torch_dtype=dtype)

        elif self.loader_type == "hf_dino":
            from transformers import ViTModel
            self.model = ViTModel.from_pretrained(self.hf_id, torch_dtype=dtype)

        elif self.loader_type == "hf_mae":
            from transformers import ViTMAEModel
            self.model = ViTMAEModel.from_pretrained(self.hf_id, torch_dtype=dtype)

        elif self.loader_type == "hf_clip":
            from transformers import CLIPVisionModel
            self.model = CLIPVisionModel.from_pretrained(self.hf_id, torch_dtype=dtype)

        elif self.loader_type == "timm_eva02":
            import timm
            self.model = timm.create_model(
                "eva02_base_patch16_clip_224.merged2b",
                pretrained=True,
                num_classes=0,
            )
            if dtype == torch.float16:
                self.model = self.model.half()

        else:
            raise ValueError(f"Unknown loader type: {self.loader_type}")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        n_params = sum(p.numel() for p in self.model.parameters())
        n_patches = (224 // self.patch_size) ** 2
        logger.info(
            "FrozenBackbone(%s): %s | %dM params | patch_size=%d | n_patches=%d",
            backbone_name, self.hf_id, n_params // 1_000_000,
            self.patch_size, n_patches,
        )

    @property
    def n_patches(self) -> int:
        return (224 // self.patch_size) ** 2

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, 224, 224) -> patch tokens (B, n_patches, 768)."""
        if self.loader_type == "hf_dinov2":
            out = self.model(pixel_values=pixel_values).last_hidden_state
            return out[:, 1:]  # drop CLS token; DINOv2 has 1 CLS + 256 patches

        elif self.loader_type == "hf_dino":
            out = self.model(pixel_values=pixel_values).last_hidden_state
            return out[:, 1:]  # drop CLS token

        elif self.loader_type == "hf_mae":
            # MAE encoder: pass mask_ratio=0 to get all patch tokens.
            # ViTMAEModel forward expects pixel_values and optional noise.
            # With default mask_ratio, some patches are masked. We want all.
            # Set noise to None and we get the encoder output for unmasked patches.
            # To get ALL patches, we use a trick: set mask_ratio attribute to 0.
            original_ratio = getattr(self.model.config, "mask_ratio", 0.75)
            self.model.config.mask_ratio = 0.0
            out = self.model(pixel_values=pixel_values)
            self.model.config.mask_ratio = original_ratio
            return out.last_hidden_state[:, 1:]  # drop CLS token

        elif self.loader_type == "hf_clip":
            # CLIPVisionModel outputs last_hidden_state: (B, 1+n_patches, 768)
            # The first token is the CLS/pooled representation.
            out = self.model(pixel_values=pixel_values).last_hidden_state
            return out[:, 1:]  # drop CLS token

        elif self.loader_type == "timm_eva02":
            # timm models with num_classes=0 return features.
            # For ViT models, forward_features returns (B, 1+n_patches, dim).
            out = self.model.forward_features(pixel_values)
            # timm ViT models typically return (B, n_patches+1, dim) with CLS first.
            if out.shape[1] == self.n_patches + 1:
                return out[:, 1:]
            return out  # some timm configs already drop CLS

        raise RuntimeError(f"Unhandled loader: {self.loader_type}")


# ---------------------------------------------------------------------------
# Normalization constants (ImageNet for all backbones)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP uses its own normalization; the HF CLIPProcessor handles it, but for
# consistency we handle it at the dataset level. CLIP's normalization is
# very close to ImageNet normalization so the difference is negligible for
# feature extraction, but we use the exact values for correctness.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_normalization(backbone_name: str) -> tuple[tuple, tuple]:
    """Return (mean, std) for the given backbone."""
    if backbone_name == "clip":
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Generic image dataset for feature extraction
# ---------------------------------------------------------------------------

class ImageListDataset(Dataset):
    """Dataset that loads images from a list of (image_path, metadata) pairs.

    Returns normalized tensors ready for backbone forward pass.
    """

    def __init__(
        self,
        items: list[dict],
        mean: tuple = IMAGENET_MEAN,
        std: tuple = IMAGENET_STD,
        out_size: int = 224,
    ) -> None:
        self.items = items
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.out_size = out_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        import math

        item = self.items[idx]
        img = Image.open(item["image_path"]).convert("RGB")

        # Square crop around bbox
        bbox = item["bbox"]  # (x, y, w, h)
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * 1.25
        x0, y0 = cx - side / 2, cy - side / 2
        crop = img.crop((
            int(round(x0)), int(round(y0)),
            int(round(x0 + side)), int(round(y0 + side)),
        ))
        crop = crop.resize((self.out_size, self.out_size), Image.BILINEAR)

        arr = np.asarray(crop, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

        # Remap keypoints to crop coordinates
        kp = np.array(item["keypoints"], dtype=np.float32)  # (K, 3)
        kp_crop = np.zeros_like(kp)
        kp_crop[:, 0] = (kp[:, 0] - x0) * (self.out_size / side)
        kp_crop[:, 1] = (kp[:, 1] - y0) * (self.out_size / side)
        kp_crop[:, 2] = kp[:, 2]
        oob = (
            (kp_crop[:, 0] < 0) | (kp_crop[:, 0] >= self.out_size)
            | (kp_crop[:, 1] < 0) | (kp_crop[:, 1] >= self.out_size)
        )
        kp_crop[oob, 2] = 0

        return {
            "pixel_values": tensor,
            "keypoints_xy": torch.from_numpy(kp_crop[:, :2]).float(),
            "vis": torch.from_numpy((kp_crop[:, 2] > 0).astype(np.float32)),
            "identity_id": item["identity_id"],
            "species_name": item["species_name"],
            "annot_id": item.get("annot_id", idx),
            "image_id": item.get("image_id", idx),
            "bbox_diag_orig": float(np.hypot(w, h)),
            "crop_x0": float(x0),
            "crop_y0": float(y0),
            "crop_side": float(side),
        }


def collate_for_extraction(batch: list[dict]) -> dict:
    out = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
    }
    out["meta"] = [
        {
            "annot_id": b["annot_id"],
            "image_id": b["image_id"],
            "species_name": b["species_name"],
            "identity_id": b["identity_id"],
            "bbox_diag_orig": b["bbox_diag_orig"],
            "crop_x0": b["crop_x0"],
            "crop_y0": b["crop_y0"],
            "crop_side": b["crop_side"],
            "keypoints_xy": b["keypoints_xy"].tolist(),
            "vis": b["vis"].tolist(),
        }
        for b in batch
    ]
    return out


# ---------------------------------------------------------------------------
# AP-10K data loading
# ---------------------------------------------------------------------------

def load_ap10k_items(
    ap10k_root: Path,
    split_file: str,
    species_filter: str = "all",
) -> list[dict]:
    """Load AP-10K annotations into a flat list of item dicts."""
    ann_path = ap10k_root / "annotations" / split_file
    with open(ann_path) as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}
    cats_by_id = {c["id"]: c for c in data["categories"]}

    # Build species filter set
    if species_filter != "all":
        filter_names = set(species_filter.split(","))
    else:
        filter_names = None

    items = []
    for ann in data["annotations"]:
        cat = cats_by_id[ann["category_id"]]
        species_name = cat["name"]
        if filter_names and species_name not in filter_names:
            continue

        img = images_by_id[ann["image_id"]]
        kp_flat = ann["keypoints"]
        if isinstance(kp_flat, str):
            kp_flat = json.loads(kp_flat)
        kp = np.asarray(kp_flat, dtype=np.float32).reshape(-1, 3)

        # Skip annotations with too few visible keypoints
        if (kp[:, 2] > 0).sum() < 4:
            continue

        items.append({
            "image_path": str(ap10k_root / "data" / img["file_name"]),
            "bbox": tuple(float(x) for x in ann["bbox"]),
            "keypoints": kp.tolist(),
            "identity_id": ann["category_id"] - 1,  # 0-based
            "species_name": species_name,
            "annot_id": ann["id"],
            "image_id": ann["image_id"],
        })

    logger.info("Loaded %d AP-10K items from %s (filter=%s)",
                len(items), split_file, species_filter)
    return items


# ---------------------------------------------------------------------------
# APT-36K data loading
# ---------------------------------------------------------------------------

def load_apt36k_items(apt36k_root: Path) -> list[dict]:
    """Load APT-36K annotations. APT-36K uses per-species annotation files."""
    apt36k_root = Path(apt36k_root)
    ann_dir = apt36k_root / "annotations"

    if not ann_dir.exists():
        logger.warning("APT-36K annotations directory not found: %s", ann_dir)
        return []

    items = []
    species_id_map = {}
    next_id = 100  # offset from AP-10K IDs

    for ann_file in sorted(ann_dir.glob("*.json")):
        try:
            with open(ann_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s", ann_file)
            continue

        # APT-36K format varies; handle COCO-style
        if "annotations" not in data:
            continue

        images_by_id = {img["id"]: img for img in data.get("images", [])}
        cats_by_id = {c["id"]: c for c in data.get("categories", [])}

        for ann in data["annotations"]:
            cat = cats_by_id.get(ann.get("category_id", 1), {})
            species_name = cat.get("name", ann_file.stem)

            if species_name not in species_id_map:
                species_id_map[species_name] = next_id
                next_id += 1

            img = images_by_id.get(ann["image_id"], {})
            if not img:
                continue

            kp_flat = ann.get("keypoints", [])
            if isinstance(kp_flat, str):
                kp_flat = json.loads(kp_flat)
            if len(kp_flat) < 6:
                continue

            kp = np.asarray(kp_flat, dtype=np.float32).reshape(-1, 3)
            if (kp[:, 2] > 0).sum() < 3:
                continue

            bbox = ann.get("bbox", [0, 0, 100, 100])
            img_path = apt36k_root / "images" / img.get("file_name", "")
            if not img_path.exists():
                img_path = apt36k_root / "data" / img.get("file_name", "")

            items.append({
                "image_path": str(img_path),
                "bbox": tuple(float(x) for x in bbox),
                "keypoints": kp.tolist(),
                "identity_id": species_id_map[species_name],
                "species_name": species_name,
                "annot_id": ann.get("id", len(items)),
                "image_id": ann["image_id"],
            })

    logger.info("Loaded %d APT-36K items across %d species",
                len(items), len(species_id_map))
    return items


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_and_cache(
    backbone: FrozenBackbone,
    dataset: ImageListDataset,
    out_dir: Path,
    batch_size: int = 16,
    num_workers: int = 2,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Extract features for the entire dataset and save to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(dataset)
    n_patches = backbone.n_patches
    hidden = backbone.hidden_dim

    logger.info("Extracting: %d items -> (%d, %d, %d) @ %s",
                n, n, n_patches, hidden, out_dir)

    feat = np.lib.format.open_memmap(
        str(out_dir / "features.npy"),
        mode="w+", dtype=np.float16, shape=(n, n_patches, hidden),
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_for_extraction,
    )

    meta_records: list[dict] = []
    written = 0
    t0 = time.time()
    last_log = t0

    for bi, batch in enumerate(loader):
        x = batch["pixel_values"].to(device, non_blocking=True)
        with torch.no_grad():
            patches = backbone(x)  # (B, n_patches, 768)
        patches = patches.to(torch.float16).cpu().numpy()
        b = patches.shape[0]
        feat[written : written + b] = patches
        meta_records.extend(batch["meta"])
        written += b

        if time.time() - last_log > 10.0:
            elapsed = time.time() - t0
            rate = written / elapsed
            eta = (n - written) / max(rate, 1e-6)
            logger.info(
                "  batch %4d/%d  written %d/%d  rate %.1f/s  ETA %.0fs",
                bi + 1, len(loader), written, n, rate, eta,
            )
            last_log = time.time()

    feat.flush()
    del feat

    (out_dir / "features.shape.json").write_text(json.dumps({
        "shape": [n, n_patches, hidden],
        "dtype": "float16",
        "filename": "features.npy",
        "n_patches_grid": 224 // backbone.patch_size,
        "backbone": backbone.backbone_name,
        "hf_id": backbone.hf_id,
        "patch_size": backbone.patch_size,
    }))
    (out_dir / "meta.json").write_text(json.dumps({
        "backbone": backbone.backbone_name,
        "hf_id": backbone.hf_id,
        "n_items": n,
        "records": meta_records,
    }))

    elapsed = time.time() - t0
    logger.info(
        "DONE [%s]: %d items in %.1fs (%.1f/s) -> %s",
        backbone.backbone_name, n, elapsed, n / max(elapsed, 1e-6), out_dir,
    )


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if prefer == "mps" or (prefer == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Precompute patch features for multiple backbones on AP-10K / APT-36K.",
    )
    p.add_argument("--ap10k-root", type=Path, default=None,
                   help="Path to AP-10K dataset root (contains data/ and annotations/).")
    p.add_argument("--apt36k-root", type=Path, default=None,
                   help="Path to APT-36K dataset root.")
    p.add_argument("--split-file", type=str, default="ap10k-train-split1.json",
                   help="AP-10K split file name.")
    p.add_argument("--species-filter", type=str, default="all",
                   help="'all' or comma-separated species names.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for cached features.")
    p.add_argument("--backbones", type=str, default="dinov2,dino,mae,clip,eva02",
                   help="Comma-separated backbone names from: " + ",".join(BACKBONE_REGISTRY))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit items per dataset for quick smoke test.")
    args = p.parse_args()

    device = select_device(args.device)
    logger.info("Device: %s", device)

    backbone_names = [b.strip() for b in args.backbones.split(",")]
    for name in backbone_names:
        if name not in BACKBONE_REGISTRY:
            raise SystemExit(f"Unknown backbone: {name}. Choose from {list(BACKBONE_REGISTRY)}")

    # Collect data items
    all_items: list[dict] = []
    dataset_label = ""

    if args.ap10k_root:
        ap10k_items = load_ap10k_items(
            args.ap10k_root, args.split_file, args.species_filter,
        )
        all_items.extend(ap10k_items)
        dataset_label += f"ap10k({len(ap10k_items)})"

    if args.apt36k_root:
        apt36k_items = load_apt36k_items(args.apt36k_root)
        all_items.extend(apt36k_items)
        dataset_label += f"+apt36k({len(apt36k_items)})"

    if not all_items:
        raise SystemExit("No data items loaded. Provide --ap10k-root and/or --apt36k-root.")

    if args.limit:
        all_items = all_items[:args.limit]
        logger.info("Limited to %d items", len(all_items))

    logger.info("Total items: %d [%s]", len(all_items), dataset_label)

    # Extract features for each backbone
    for backbone_name in backbone_names:
        logger.info("=" * 60)
        logger.info("Backbone: %s", backbone_name)
        logger.info("=" * 60)

        backbone = FrozenBackbone(backbone_name).to(device)

        mean, std = get_normalization(backbone_name)
        dataset = ImageListDataset(all_items, mean=mean, std=std)

        cache_dir = args.out_dir / backbone_name
        extract_and_cache(
            backbone, dataset, cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        # Free GPU memory before next backbone
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("All backbones complete. Output: %s", args.out_dir)


if __name__ == "__main__":
    main()
