from dotenv import load_dotenv
import os
import pathlib

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Print .env file path and existence
env_path = pathlib.Path('.env')
print(f".env file exists: {env_path.exists()}")

# Try to read .env file directly
try:
    with open('.env', 'r') as f:
        print("Raw .env contents:", f.read())
except Exception as e:
    print(f"Error reading .env: {e}")

# Load environment variables
load_dotenv(override=True)  # Add override=True to force reload

# Get and print API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"First few characters of loaded key: {api_key[:7]}...")