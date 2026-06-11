from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import logging

logger = logging.getLogger("ImageCropper")

try:
    from PIL import Image, ImageEnhance, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning('[Cropper] PIL not available')

try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False



RESOLUTION_BUCKETS = [
    (768, 1344), (832, 1216), (896, 1152), (1024, 1024), 
    (1152, 896), (1216, 832), (1344, 768),
    (512, 768), (768, 512), (512, 512),
    (640, 640), (704, 512), (512, 704)
]

ASPECT_RATIOS = {
    '1:1': (1, 1), '4:3': (4, 3), '3:4': (3, 4), 
    '16:9': (16, 9), '9:16': (9, 16), '3:2': (3, 2), '2:3': (2, 3)
}

def find_closest_resolution(image_ratio: float, resolutions: List[Tuple[int, int]]) -> Tuple[int, int]:
    if not resolutions:
        return (1024, 1024)
    sorted_res = sorted(resolutions, key=lambda r: abs(image_ratio - (r[0] / r[1])))
    return sorted_res[0]

class ImageCropper:
    def __init__(self, resolutions: List[Tuple[int, int]] = None):
        self.resolutions = resolutions or RESOLUTION_BUCKETS
        self.face_cascade = None
        if CV_AVAILABLE:
            try:
                # Load Haar Cascade for face detection
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if not Path(cascade_path).exists():
                     # Fallback or try-except capture below
                     pass
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                     logger.warning("Face cascade failed to load (empty).")
                     self.face_cascade = None
            except Exception as e:
                logger.warning(f"Failed to load face cascade: {e}")

    def detect_faces(self, pil_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image, returns list of (x, y, w, h)"""
        if not CV_AVAILABLE or self.face_cascade is None:
            return []
        
        try:
            # Convert PIL to OpenCV (Grayscale for speed)
            cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(cv_img, 1.1, 4)
            return faces.tolist()
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []

    def get_crop_region(self, img_w: int, img_h: int, target_w: int, target_h: int, faces: List[Tuple[int, int, int, int]] = None) -> Tuple[int, int, int, int]:
        """Calculate crop coordinates. If faces are provided, try to center around them."""
        scale = max(target_w / img_w, target_h / img_h)
        sw = int(img_w * scale)
        sh = int(img_h * scale)
        
        # Center of the crop in scaled coordinates
        center_x = sw // 2
        center_y = sh // 2
        
        if faces:
            # Calculate collective center of all faces
            fx_sum = 0
            fy_sum = 0
            for (x, y, w, h) in faces:
                fx_sum += (x + w // 2)
                fy_sum += (y + h // 2)
            
            # Map face center to scaled coordinates
            center_x = int((fx_sum / len(faces)) * scale)
            center_y = int((fy_sum / len(faces)) * scale)
            
        # Define bounds
        left = max(0, min(center_x - target_w // 2, sw - target_w))
        top = max(0, min(center_y - target_h // 2, sh - target_h))
        right = left + target_w
        bottom = top + target_h
        
        return (left, top, right, bottom, sw, sh)

    def enhance_image(self, image: Image.Image, config: Dict[str, Any]) -> Image.Image:
        """Apply color enhancements and normalization"""
        # Brightness
        brightness = config.get('brightness', 1.0)
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
            
        # Contrast
        contrast = config.get('contrast', 1.0)
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)
            
        # Color (Saturation)
        color = config.get('saturation', 1.0)
        if color != 1.0:
            image = ImageEnhance.Color(image).enhance(color)
            
        # Histogram Equalization / Normalization
        if config.get('equalize', False):
            image = ImageOps.equalize(image)
            
        if config.get('auto_contrast', False):
            image = ImageOps.autocontrast(image)
            
        return image

    def process_single(self, image: Image.Image, config: Dict[str, Any]) -> Image.Image:
        mode = config.get('mode', 'bucket')
        auto_crop = config.get('auto_crop', False)
        target_size = config.get('target_size', 1024)
        
        # Flips (Apply before crop/scale to maintain landmarks if we held any)
        if config.get('flip_horizontal', False):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if config.get('flip_vertical', False):
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            
        # Rotation
        rotate = config.get('rotate', 0)
        if rotate != 0:
            # We use expand=True to avoid cutting corners, but for training datasets 
            # usually 90/180/270 is used which doesn't change aspect ratio significantly or at all
            image = image.rotate(rotate, expand=True)

        width, height = image.size
        ratio = width / height
        
        if mode == 'bucket':
            target_w, target_h = find_closest_resolution(ratio, self.resolutions)
        else:
            ratio_str = config.get('ratio', '1:1')
            rw, rh = ASPECT_RATIOS.get(ratio_str, (1, 1))
            r = rw / rh
            if r >= 1:
                target_w = target_size
                target_h = int(target_size / r)
            else:
                target_h = target_size
                target_w = int(target_size * r)
        
        if width == target_w and height == target_h:
            return image

        faces = self.detect_faces(image) if auto_crop else []
        left, top, right, bottom, sw, sh = self.get_crop_region(width, height, target_w, target_h, faces)
        
        # Scale and crop
        resized = image.resize((sw, sh), resample=Image.Resampling.LANCZOS)
        cropped = resized.crop((left, top, right, bottom))
        
        # Apply enhancements
        return self.enhance_image(cropped, config)

    def process_batch(self, images: List[Image.Image], filenames: List[str], config: Dict[str, Any], progress_callback=None) -> List[Dict[str, Any]]:
        results = []
        total = len(images)
        for i, (image, filename) in enumerate(zip(images, filenames)):
            original_size = image.size
            processed = self.process_single(image, config)
            results.append({
                'filename': filename, 
                'image': processed, 
                'original_size': original_size, 
                'new_size': processed.size
            })
            if progress_callback:
                progress_callback(i + 1, total)
        return results

def get_available_ratios() -> List[str]:
    return list(ASPECT_RATIOS.keys())

def get_resolution_buckets() -> List[Tuple[int, int]]:
    return RESOLUTION_BUCKETS.copy()
