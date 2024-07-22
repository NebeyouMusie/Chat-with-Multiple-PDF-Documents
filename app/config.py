from dotenv import load_dotenv
import os

# load configuration fot environment variables
def load_config():
    return load_dotenv()

def get_google_api_key():
    return os.getenv("GOOGLE_API_KEY")