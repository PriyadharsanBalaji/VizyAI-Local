import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import Union, Optional

class ImageUtils:
    """
    Utilities for image processing and encoding
    """
    
    @staticmethod
    def encode_image_to_base64(image_path: Union[str, Path]) -> str:
        """Encode image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def pil_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        pil_image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    @staticmethod
    def resize_image(image_path: Union[str, Path], 
                    max_size: tuple = (1024, 1024),
                    output_path: Optional[str] = None) -> str:
        """Resize image to fit within max_size"""
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        if output_path is None:
            output_path = image_path
        
        img.save(output_path, optimize=True, quality=85)
        return str(output_path)
    
    @staticmethod
    def compress_image(image_path: Union[str, Path], 
                      quality: int = 85,
                      max_size_kb: int = 500) -> str:
        """Compress image to reduce file size"""
        img = Image.open(image_path)
        
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        
        current_quality = quality
        output_path = Path(image_path).with_suffix('.jpg')
        
        while current_quality > 20:
            img.save(output_path, 'JPEG', quality=current_quality, optimize=True)
            
            if output_path.stat().st_size / 1024 <= max_size_kb:
                break
            
            current_quality -= 5
        
        return str(output_path)
    
    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> dict:
        """Get image metadata"""
        img = Image.open(image_path)
        file_size = Path(image_path).stat().st_size / 1024
        
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_kb": round(file_size, 2)
        }
