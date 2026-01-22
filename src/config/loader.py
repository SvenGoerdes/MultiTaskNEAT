from pathlib import Path

# Dieser Code ermittelt den Root-Pfad deines Projekts
# Ausgehend von src/config/loader.py geht er zwei Ebenen hoch
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Jetzt kannst du Pfade immer vom Root aus bauen
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        # Lade-Logik hier
        pass