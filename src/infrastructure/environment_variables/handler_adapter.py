import os
from dotenv import load_dotenv


class CredentialHandler:
    def __init__(self, env_folder):
        #Loading .env
        current_directory = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(current_directory, '..', env_folder, '.env')
        load_dotenv(override=True, dotenv_path=env_path)
        
        for key, value in os.environ.items():
            if not hasattr(self, key):
                setattr(self, key, value)
            