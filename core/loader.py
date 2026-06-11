import os
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import logging

logger = logging.getLogger("BatchLoader")
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning('[WARNING] PyTorch not installed. BatchLoader will have limited functionality.')

class BatchLoader:

    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def load(self, image_directory, limit=0, resize_method='Resize to First'):
        if not image_directory or not os.path.exists(image_directory):
            logger.error(f'[BatchLoader] Invalid directory: {image_directory}')
            return ([], [])  # Return empty list, not placeholder tensor

        input_path = Path(image_directory)
        files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in self.supported_formats]
        files.sort()
        if limit > 0:
            files = files[:limit]
        
        if not files:
            logger.info('[BatchLoader] No images found.')
            return ([], [])
        image_list = []
        filename_list = []
        target_w, target_h = (0, 0)
        for i, fpath in enumerate(files):
            try:
                with Image.open(fpath) as img_raw:
                     # Copy or process inside context
                     img = ImageOps.exif_transpose(img_raw)
                     if img.mode != 'RGB':
                         img = img.convert('RGB')
                     # Need to deep copy if we close file? Image.open is lazy.
                     # load() forces loading.
                     img.load()
                     
                     if i == 0:
                         target_w, target_h = img.size
                     
                     if img.size != (target_w, target_h):
                         if resize_method == 'Resize to First':
                             img = img.resize((target_w, target_h), Image.LANCZOS)
                         elif resize_method == 'Center Crop to First':
                             curr_w, curr_h = img.size
                             left = (curr_w - target_w) // 2
                             top = (curr_h - target_h) // 2
                             if curr_w < target_w or curr_h < target_h:
                                 img = img.resize((target_w, target_h), Image.LANCZOS)
                             else:
                                 img = img.crop((left, top, left + target_w, top + target_h))
                         elif resize_method == 'None (Strict)':
                             logger.warning(f'[BatchLoader] Skipping {fpath.name}, size mismatch')
                             continue
                     
                     img_np = np.array(img).astype(np.float32) / 255.0
                     if TORCH_AVAILABLE:
                         image_list.append(torch.from_numpy(img_np))
                     else:
                         image_list.append(img_np)
                     filename_list.append(fpath.name)
            except Exception as e:
                logger.error(f'[BatchLoader] Error loading {fpath}: {e}')
        if not image_list:
            return ([], [])
        if TORCH_AVAILABLE:
            batch_tensor = torch.stack(image_list)
        else:
            batch_tensor = np.stack(image_list)
        return (batch_tensor, filename_list)