from huggingface_hub import snapshot_download
import os

from huggingface_hub import login

def load_properties(filepath: str) -> dict:
    properties = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    properties[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"File Not Found")
    except Exception as e:
        print(f"e")
    return properties

def get_huggingface_access_token(config_filepath: str = 'config.properties') -> str:
    config = load_properties(config_filepath)
    token = config.get('huggingface.access.token')

    if token:
        return token
    else:
        return None

# huggingface_key = get_huggingface_access_token("../../../../../models/huggingface.properties")
# login(huggingface_key)

def huggingface_login_script(hf_token: str):

    if not hf_token:
        print("no hf_token")
        return

    try:
        login(token=hf_token)
    except Exception as e:
        print(f"{e}")

# huggingface_login_script(huggingface_key)

model_id = "openai/clip-vit-base-patch32"
local_dir = "./clip-vit-base-patch32" 

os.makedirs(local_dir, exist_ok=True)

print(f"Start Downloading...")

snapshot_download(repo_id=model_id, local_dir=local_dir,
                  allow_patterns=["*.json", "*.bin", "*.txt", "*.model"], 
                  ignore_patterns=["*.safetensors"], 
                  local_dir_use_symlinks=False) 

print("Downloaded!")