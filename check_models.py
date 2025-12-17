"""
Check available Gemini models using the Google GenAI API.
"""

import os

# Read API key from .env file manually
api_key = None
try:
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('GOOGLE_API_KEY='):
                api_key = line.strip().split('=', 1)[1]
                break
except FileNotFoundError:
    pass

if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in .env file")
    exit(1)

print("=" * 80)
print("CHECKING AVAILABLE GEMINI MODELS")
print("=" * 80)
print(f"\nAPI Key: {api_key[:10]}...{api_key[-4:]}\n")

try:
    import google.generativeai as genai

    # Configure the API
    genai.configure(api_key=api_key)

    print("Fetching available models...\n")

    # List all available models
    models = genai.list_models()

    print(f"Found {len(list(models))} models:\n")
    print("-" * 80)

    # Fetch models again to iterate (generator is consumed)
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"[OK] Model: {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Supported Methods: {', '.join(model.supported_generation_methods)}")
            print(f"  Description: {model.description[:100]}...")
            print("-" * 80)

    print("\n" + "=" * 80)
    print("RECOMMENDED MODEL NAMES TO USE:")
    print("=" * 80)

    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            # Extract just the model name part (remove 'models/' prefix)
            model_name = model.name.replace('models/', '')
            print(f"  {model_name}")

except ImportError:
    print("ERROR: google-generativeai package not installed")
    print("Install it with: pip install google-generativeai")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
