import os
import logging
import joblib

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def setup_logging(level=logging.INFO, log_format=None):
    if log_format is None:
        log_format = "[%(asctime)s] %(levelname)s — %(name)s — %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("PhishingDetection")
    logger.setLevel(level)
    return logger

def save_artifact(obj, path):
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)
    print(f"Artifact disimpan ke: {path}")
    return path
    
def load_artifact(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact tidak ditemukan: {path}")
    obj = joblib.load(path)
    print(f"Artifact dimuat dari: {path}")
    return obj