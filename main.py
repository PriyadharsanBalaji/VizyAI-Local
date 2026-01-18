#!/usr/bin/env python3
"""
AI-Powered Visualization & Insights Tool - Ollama Local Version
Main entry point for the application
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from utils.logger import LoggerSetup

def check_ollama_running():
    """Check if Ollama is running"""
    import requests
    try:
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main entry point"""
    
    # Setup logging
    logger = LoggerSetup.setup_logger()
    logger.info("=" * 60)
    logger.info("AI-Powered Visualization & Insights Tool - Ollama Local")
    logger.info("=" * 60)
    
    # Check Ollama
    logger.info(f"Checking Ollama service at {Config.OLLAMA_BASE_URL}...")
    if not check_ollama_running():
        logger.error("Ollama is not running!")
        logger.error(f"Please start Ollama or check URL: {Config.OLLAMA_BASE_URL}")
        logger.error("Windows: Ollama should auto-start. Check system tray.")
        logger.error("Manual start: Run 'ollama serve' in terminal")
        print("\n" + "="*60)
        print("ERROR: Ollama not running!")
        print("="*60)
        print(f"Expected at: {Config.OLLAMA_BASE_URL}")
        print("\nTo fix:")
        print("1. Check if Ollama is running (system tray icon)")
        print("2. Or run: ollama serve")
        print("3. Verify model installed: ollama list")
        print(f"4. Pull model if needed: ollama pull {Config.OLLAMA_MODEL}")
        print("="*60)
        sys.exit(1)
    
    logger.info(f"Configuration loaded")
    logger.info(f"Ollama URL: {Config.OLLAMA_BASE_URL}")
    logger.info(f"Model: {Config.OLLAMA_MODEL}")
    logger.info(f"Cache enabled: {Config.CACHE_GRAPHS}")
    logger.info("No rate limits - running locally!")
    
    # Launch Streamlit app
    streamlit_app = project_root / "streamlit_app" / "app.py"
    
    logger.info(f"Launching Streamlit app...")
    logger.info(f"App location: {streamlit_app}")
    
    try:
        subprocess.run([
            "streamlit", "run",
            str(streamlit_app),
            "--server.headless", "true",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        logger.info("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Error launching app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
