import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Ollama Configuration (CHANGED from Gemini)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision:11b")
    
    # No rate limiting needed - it's local! (REMOVED rate limit configs)
    OLLAMA_TIMEOUT = 300  # 5 minutes max per request
    MAX_RETRIES = 3
    
    # Graph generation settings (SAME)
    DEFAULT_K_VALUE = 5
    MAX_K_VALUE = 15
    MIN_K_VALUE = 1
    
    # Cache settings (SAME)
    CACHE_DIR = "cache"
    CACHE_GRAPHS = True
    CACHE_LLM_RESPONSES = True
    
    # Visualization settings (SAME)
    FIGURE_DPI = 100
    FIGURE_FORMAT = "png"
    MAX_GRAPHS_GENERATE = 30
    
    # Data limits (SAME)
    MAX_UPLOAD_SIZE_MB = 200
    SAMPLE_ROWS_FOR_LLM = 10
