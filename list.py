import requests

GEMINI_API_KEY = "AIzaSyDLE9wfQuSfXxvKGtA1sLWhgePW_bKfBVU"

def list_gemini_models():
    """List all available Gemini models"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            models = result.get("models", [])
            
            print("\n" + "=" * 80)
            print("📋 AVAILABLE GEMINI MODELS")
            print("=" * 80)
            
            for model in models:
                name = model.get("name", "Unknown")
                display_name = model.get("displayName", "")
                description = model.get("description", "")
                supported_methods = model.get("supportedGenerationMethods", [])
                
                print(f"\n🔹 Model: {name}")
                print(f"   Display Name: {display_name}")
                print(f"   Description: {description[:100]}..." if len(description) > 100 else f"   Description: {description}")
                print(f"   Supported Methods: {', '.join(supported_methods)}")
            
            print("\n" + "=" * 80)
            print(f"✅ Total models found: {len(models)}")
            print("=" * 80 + "\n")
            
            return models
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return []

if __name__ == "__main__":
    list_gemini_models()
