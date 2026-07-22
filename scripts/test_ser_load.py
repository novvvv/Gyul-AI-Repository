import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.services.ser_service import SERService


def main() -> None:
    service = SERService()
    service.load()
    labels = getattr(service.model.config, "id2label", {})
    print("SER model loaded:", service.is_loaded)
    print("Label count:", len(labels))
    print("Labels:", labels)


if __name__ == "__main__":
    main()
