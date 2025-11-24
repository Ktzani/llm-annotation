import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infrastructure.environment_variables.handler_adapter import CredentialHandler

common_prefix_s3 = "s3://"
current_path = os.getcwd()
credentials_handler = CredentialHandler(env_folder=f"/home/catiza/Documentos/Git/science-gt589-forecaster/src/infrastructure/environment_variables")