import os
from pathlib import Path
from PIL import Image
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import logging
logger = logging.getLogger("BatchExporter")

class BatchExporter:

    def export(self, images, filenames, tags_dict, output_dir, config):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        format_ext = config.get('format', 'png')
        quality = config.get('quality', 95)
        save_tags = config.get('save_tags', True)
        success_count = 0
        for i, (img_tensor, filename) in enumerate(zip(images, filenames)):
            try:
                if TORCH_AVAILABLE and torch.is_tensor(img_tensor):
                    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                else:
                    img_np = (img_tensor * 255).astype(np.uint8)
                img = Image.fromarray(img_np)
                base_name = Path(filename).stem
                output_file = output_path / f'{base_name}.{format_ext}'
                if format_ext.lower() in ['jpg', 'jpeg']:
                    img.save(output_file, 'JPEG', quality=quality)
                else:
                    img.save(output_file, 'PNG')
                if save_tags and filename in tags_dict:
                    tag_file = output_path / f'{base_name}.txt'
                    with open(tag_file, 'w', encoding='utf-8') as f:
                        f.write(tags_dict[filename])
                success_count += 1
            except Exception as e:
                logger.error(f'[Exporter] Error exporting {filename}: {e}')
        logger.info(f'[Exporter] Successfully exported {success_count}/{len(images)} images')
        return success_count