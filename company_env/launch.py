"""Start the simulated company environment (portal) in a background thread."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

from werkzeug.serving import make_server

from .portal.app import create_app

ENV_ROOT = Path(__file__).resolve().parent
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def copy_company(dest: Path) -> Path:
    """A private copy of the company file share + config (so a run never modifies the repository's data)."""
    dest = Path(dest)
    for part in ("fs", "config"):
        shutil.copytree(ENV_ROOT / part, dest / part, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "expense_summary_*"))
    return dest


@dataclass
class RunningPortal:
    url: str
    db_path: str
    audit_token: str
    app: object
    _server: object
    _thread: Thread

    def set_faults(self, **faults) -> None:
        self.app.config["FAULTS"].update(faults)
        self.app.config["FIRED"].clear()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()      # release the port so a resumed run can rebind it
        self._thread.join(timeout=5)


def start_portal(faults: dict | None = None, port: int = 0, db_path: str | None = None,
                 today: str = "2026-09-20") -> RunningPortal:
    vault = json.loads((ENV_ROOT / "config" / "vault.json").read_text(encoding="utf-8"))
    creds = vault["expense_portal"]
    users = {creds["username"]: creds["password"], "rahul.verma": "pw-rahul-demo"}
    audit_token = vault["expense_portal_audit"]["token"]
    if db_path is None:
        fd, db_path = tempfile.mkstemp(prefix="expense_portal_", suffix=".db")
        os.close(fd)
        os.unlink(db_path)
    app = create_app(db_path, users, audit_token, faults=faults, today=today)
    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return RunningPortal(url=f"http://127.0.0.1:{server.server_port}", db_path=db_path,
                         audit_token=audit_token, app=app, _server=server, _thread=thread)
