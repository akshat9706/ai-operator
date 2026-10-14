import os
import sys
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env reader: lines like KEY=value. Existing environment variables win. Only used when the program is
    started from the command line (not by the tests), so a key in .env can never leak into an automated test."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


load_dotenv()

from .cli import main  # noqa: E402

sys.exit(main())
