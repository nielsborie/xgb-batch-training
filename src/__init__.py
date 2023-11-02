import os
from pathlib import Path

project_root_path = Path(__file__).parent.parent
data_dir = os.path.join(project_root_path, "data")
reports_dir = os.path.join(project_root_path, "reports")
models_dir = os.path.join(project_root_path, "models")