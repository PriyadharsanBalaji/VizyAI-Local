import ollama
from PIL import Image
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config
import time

class OllamaClient:
    """
    Wrapper for Ollama vision model - LOCAL, NO RATE LIMITS!
    Replaces GeminiClient with exact same interface
    """
    
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = Config.OLLAMA_MODEL
        
        self.model_name = model_name
        self.base_url = Config.OLLAMA_BASE_URL
        
        # Initialize Ollama client
        try:
            self.client = ollama.Client(host=self.base_url)
            # Verify model exists
            models = self.client.list()
            # Fix: Handle different response structures
            if isinstance(models, dict) and 'models' in models:
                available = [m.get('name', m.get('model', '')) for m in models['models']]
            else:
                available = []
            
            if available and model_name not in available:
                print(f"⚠️  Model {model_name} not found!")
                print(f"Available: {available}")
                print(f"Run: ollama pull {model_name}")
        except Exception as e:
            print(f"⚠️  Ollama connection error: {e}")
            print("Is Ollama running? Check: http://localhost:11434")
    
    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def generate_content(self, prompt, image_path=None, wait_if_limited=True):
        """
        Generate content with optional image
        
        Args:
            prompt: Text prompt
            image_path: Optional path to image file
            wait_if_limited: Ignored for local model (kept for compatibility)
            
        Returns:
            Generated text response
        """
        try:
            if image_path:
                # Vision request with image
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }]
                )
            else:
                # Text-only request
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }]
                )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"Error generating content: {e}")
            raise
    
    def get_rate_limit_status(self):
        """
        Get status - returns local status (no rate limits!)
        Kept for compatibility with original code
        """
        try:
            models = self.client.list()
            return {
                'rpm_used': 0,
                'rpm_limit': 999999,  # No limit!
                'tpm_used': 0,
                'tpm_limit': 999999,
                'rpd_used': 0,
                'rpd_limit': 999999,
                'status': 'local_running',
                'model': self.model_name,
                'rate_limited': False  # FIX: Added missing key
            }
        except Exception as e:
            return {
                'rpm_used': 0,
                'rpm_limit': 0,
                'tpm_used': 0,
                'tpm_limit': 0,
                'rpd_used': 0,
                'rpd_limit': 0,
                'status': 'error',
                'model': self.model_name,
                'rate_limited': False,  # FIX: Added missing key
                'error': str(e)
            }
