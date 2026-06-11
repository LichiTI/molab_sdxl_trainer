import os
import numpy as np
import math
from PIL import Image
import logging

logger = logging.getLogger("UpscalerEngine")

try:
    import torch
    from .architecture import RRDBNet
    from .tensorrt_acceleration import rrdb_state_from_checkpoint
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not found. Upscaler will operate in NCNN-only mode.")
    RRDBNet = None
    rrdb_state_from_checkpoint = None

class UpscalerEngine:
    def __init__(self, device=None):
        if device is not None and device not in ["cpu", "cuda"]:
            logger.warning(f"Invalid device '{device}', falling back to auto-detection.")
            device = None # Trigger auto-detect below

        if device is None:
            if HAS_TORCH and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.model = None
        self.model_ncnn = None
        self.current_model_path = None
        self.scale = 4
        self.tile_size = 512
        self.tile_pad = 32
        
        self.engine_type = "torch" # or "ncnn"

    def load_model(self, model_path: str, scale: int = 4):
        """Load a upscaling model (supports .pth and .bin)."""
        if self.current_model_path == model_path and (self.model is not None or self.model_ncnn is not None):
            return

        logger.info(f"Loading upscaler model: {model_path}")
        
        if model_path.endswith(".bin"):
             self._load_ncnn_model(model_path, scale)
        elif model_path.endswith(".pth"):
             self._load_torch_model(model_path, scale)
        else:
             raise ValueError(f"Unsupported model format: {model_path}")
             
        self.current_model_path = model_path
        
    def _load_ncnn_model(self, bin_path: str, scale: int):
        """Load NCNN model."""
        try:
            import ncnn
        except ImportError:
            raise ImportError("NCNN Runtime not found. Please install 'NCNN Runtime' from the Extensions Center.")

        param_path = bin_path.replace(".bin", ".param")
        if not os.path.exists(param_path):
             raise FileNotFoundError(f"Missing parameter file: {param_path}")

        self.model_ncnn = ncnn.Net()
        # Load param first, then binary
        self.model_ncnn.load_param(str(param_path))
        self.model_ncnn.load_model(str(bin_path))
        
        self.model = None # Clear torch model
        self.engine_type = "ncnn"
        self.scale = scale
        self.model_ncnn.opt.use_vulkan_compute = True # Enable GPU if available
        logger.info(f"Loaded NCNN model: {bin_path} (Scale: {scale})")

    def _load_torch_model(self, model_path: str, scale: int = 4):
        """Load a PyTorch model (.pth)"""
        if not HAS_TORCH or RRDBNet is None:
             raise ImportError("PyTorch not installed. Cannot load .pth models.")

        if self.current_model_path == model_path and self.model is not None:
            return

        logger.info(f"Loading upscaler model: {model_path}")
        
        # Initialize RRDBNet (Standard 4x architecture)
        # Most RealESRGAN/Ultrasharp models use: 
        # in_nc=3, out_nc=3, num_feat=64, num_block=23, num_grow_ch=32
        self.model = RRDBNet(in_nc=3, out_nc=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        
        # Load weights
        loadnet = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        
        if rrdb_state_from_checkpoint is None:
            raise ImportError("RRDB state converter not available.")
        self.model.load_state_dict(rrdb_state_from_checkpoint(loadnet), strict=True)
            
        self.model.eval()
        self.model.to(self.device)
        self.model_ncnn = None # Clear NCNN model
        self.engine_type = "torch" 
        self.scale = scale
        logger.info("Model loaded successfully.")

    def upscale_image(self, img_path: str, output_path: str, format: str = "png") -> str:
        """Upscale an image and save to disk."""
        if self.engine_type == "ncnn":
             return self._upscale_ncnn(img_path, output_path, format)
             
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Load image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        with Image.open(img_path) as src_img:
            img = src_img.convert('RGB')
            img_np = np.array(img)
            
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(self.device)

        # Inference with tiling
        with torch.no_grad():
            output_t = self.tile_process(img_t, self.tile_size, self.tile_pad)

        # Post-process
        # Post-process
        output = output_t.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        # CHW -> HWC (Standard RGB for PIL)
        output = np.transpose(output, (1, 2, 0)) 
        
        # Standard conversion handling
        output = (output * 255.0).round().astype(np.uint8)
        
        # Save
        out_img = Image.fromarray(output)
        
        # Ensure output directory exists (if not current dir)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        out_img.save(output_path, format=format)
        
        return output_path

    def _upscale_ncnn(self, img_path: str, output_path: str, format: str = "png") -> str:
        """Upscale using NCNN engine."""
        try:
            import ncnn
        except ImportError:
            raise RuntimeError("NCNN module not found during inference.")
        import cv2
        
        # Load image with cv2 (NCNN style)
        img = cv2.imread(img_path)
        if img is None:
             raise RuntimeError(f"Failed to read image: {img_path}")
             
        h, w, c = img.shape
        
        # Helper to process tiles with ncnn
        # Note: NCNN handles tiling internally? No, we must implementation it or just feed whole image.
        # NCNN on GPU is usually memory efficient. Let's try simple full inference first.
        # If OOM, we can add tiling later.
        
        # Create input/output blobs
        in_mat = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_BGR, w, h)
        
        ex = self.model_ncnn.create_extractor()
        ex.input("input", in_mat)
        
        ret, out_mat = ex.extract("output")
        if ret != 0:
             raise RuntimeError(f"NCNN inference failed with code {ret}")
             
        # Convert output back to numpy
        # out_mat: c, h, w
        out_h = out_mat.h
        out_w = out_mat.w
        
        # Get output bytes and shape
        out_img_np = np.array(out_mat) # This usually gives a byte array or list? 
        # NCNN python bindings are tricky. Let's use more standard approach for safety
        # Current ncnn-python approach:
        out_bytes = bytearray(out_mat)
        out_img_np = np.frombuffer(out_bytes, dtype=np.uint8).reshape((out_h, out_w, 3))
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, out_img_np)
        
        return output_path

    def tile_process(self, img, tile_size, tile_pad):
        """Process image in tiles to save VRAM."""
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # Start with cpu to save vram
        output = torch.zeros(output_shape, device=self.device)
        
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                # Extract tile coordinates
                ofs_y = y * tile_size
                ofs_x = x * tile_size
                
                # Input tile boundaries
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)

                # Input tile boundaries with padding
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)

                # Input tile dimensions w/ padding
                input_tile_width = input_end_x_pad - input_start_x_pad
                input_tile_height = input_end_y_pad - input_start_y_pad

                # Extract input tile
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # Run inference
                with torch.no_grad():
                    output_tile = self.model(input_tile)

                # Output tile boundaries (mapped from input w/o padding)
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale

                # Output padding offsets
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * self.scale
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * self.scale

                # Place into output buffer
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

        return output
