"""
Test Ollama connection and vision capabilities
"""
from llm_agents.ollama_client import OllamaClient
from config import Config
import sys
import os

def test_ollama():
    """Test if Ollama is working"""
    
    print("=" * 60)
    print("Testing Ollama Local Vision Model")
    print("=" * 60)
    
    print(f"\nOllama URL: {Config.OLLAMA_BASE_URL}")
    print(f"Model: {Config.OLLAMA_MODEL}")
    
    try:
        client = OllamaClient()
        
        # Test text-only
        print("\n1. Testing text generation...")
        response = client.generate_content("Say 'Ollama is working!' in one sentence.")
        print(f"✅ Response: {response}")
        
        # Get status
        status = client.get_rate_limit_status()
        print(f"\n📊 Status:")
        print(f"  Model: {status.get('model', 'Unknown')}")
        print(f"  Status: {status.get('status', 'Unknown')}")
        print(f"  Rate Limited: {status.get('rate_limited', False)}")
        
        if status.get('error'):
            print(f"  Warning: {status['error']}")
        
        # Test with sample image (if exists)
        if os.path.exists("graphs"):
            graph_files = [f for f in os.listdir("graphs") if f.endswith('.png')]
            if graph_files:
                print(f"\n2. Testing vision with image: {graph_files[0]}")
                img_path = os.path.join("graphs", graph_files[0])
                response = client.generate_content(
                    "Describe what you see in this graph in one sentence.",
                    image_path=img_path
                )
                print(f"✅ Vision Response: {response}")
            else:
                print("\n2. No graphs found to test vision. Run app first to generate graphs.")
        else:
            print("\n2. No graphs folder found. Run app first to generate graphs.")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nOllama is ready to use!")
        print(f"Run: python main.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? Check system tray or run 'ollama serve'")
        print(f"2. Is model installed? Run: ollama pull {Config.OLLAMA_MODEL}")
        print("3. Check Ollama URL in .env file")
        print("\nQuick check:")
        print(f"  curl {Config.OLLAMA_BASE_URL}/api/tags")
        sys.exit(1)

if __name__ == "__main__":
    test_ollama()
