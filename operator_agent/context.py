"""Company context: sandboxed file share, knowledge search, systems registry, credential vault."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .permissions import render_placeholders


class ToolError(Exception):
    """A problem the model can act on. `kind` maps to ToolResult.error_kind."""

    def __init__(self, message: str, kind: str = "tool_error"):
        super().__init__(message)
        self.kind = kind


class Sandbox:
    """The file share the operator may see. Paths are relative to the root; escapes are refused."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def resolve(self, rel: str) -> Path:
        rel = (rel or ".").lstrip("/")
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"Path '{rel}' is outside the company file share.", "denied")
        return p

    def rel(self, p: Path) -> str:
        return p.resolve().relative_to(self.root).as_posix()


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "is", "are", "how", "do", "i", "my",
         "what", "with", "be", "it", "this", "that"}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class KnowledgeBase:
    """BM25 search over the markdown documents in knowledge/ (chunked by heading)."""

    def __init__(self, sandbox: Sandbox, subdir: str = "knowledge"):
        self.chunks: list[dict] = []
        base = sandbox.root / subdir
        for path in sorted(base.rglob("*.md")) if base.exists() else []:
            heading, buf = path.stem, []
            def flush():
                text = "\n".join(buf).strip()
                if text:
                    self.chunks.append({"path": sandbox.rel(path), "heading": heading, "text": text,
                                        "tokens": _tokens(heading + " " + text)})
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    flush()
                    heading, buf = line.lstrip("# ").strip(), []
                else:
                    buf.append(line)
            flush()
        n = max(len(self.chunks), 1)
        self.avg = sum(len(c["tokens"]) for c in self.chunks) / n
        df: Counter = Counter()
        for c in self.chunks:
            df.update(set(c["tokens"]))
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = _tokens(query)
        scored = []
        for c in self.chunks:
            tf = Counter(c["tokens"])
            score = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    score += self.idf.get(t, 0) * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len(c["tokens"]) / self.avg))
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return [{"path": c["path"], "heading": c["heading"], "score": round(s, 2),
                 "snippet": c["text"][:320].replace("\n", " ")} for s, c in scored[:k]]


@dataclass
class Systems:
    """Registry of company systems plus the credential vault. Only harness code reads the vault."""
    systems: dict[str, dict]
    _vault: dict[str, dict] = field(repr=False, default_factory=dict)

    @classmethod
    def load(cls, config_dir: Path, placeholders: dict[str, str]) -> "Systems":
        cfg = json.loads(render_placeholders((Path(config_dir) / "systems.json").read_text(encoding="utf-8"), placeholders))
        vault = json.loads((Path(config_dir) / "vault.json").read_text(encoding="utf-8"))
        return cls(cfg, vault)

    def credential(self, name: str) -> dict:
        if name not in self._vault:
            raise ToolError(f"No credential '{name}' in the vault.", "tool_error")
        return self._vault[name]

    def system_for_url(self, url: str) -> str | None:
        host = urlparse(url)
        for name, s in self.systems.items():
            b = urlparse(s["base_url"])
            if (host.scheme, host.netloc) == (b.scheme, b.netloc):
                return name
        return None

    def describe(self) -> list[dict]:
        out = []
        for n, s in self.systems.items():
            item = {"name": n, "description": s.get("description", ""), "base_url": s["base_url"],
                    "sign_in": f"call browser_login with system='{n}'"}
            if s.get("audit"):
                pre = s["audit"]["path_prefix"]
                item["for_verification_claims"] = (f"json_check paths must start with {pre} (for example "
                                                   f"{pre}/reports/<id>.json). You cannot open these yourself; only "
                                                   "the verifier can.")
            out.append(item)
        return out


def resource_key(systems: Systems, url: str) -> str:
    """Stable identity of the thing a URL refers to: '<system>:/reports/3' for /reports/3/lines/new."""
    u = urlparse(url)
    segs = [s for s in u.path.split("/") if s]
    if len(segs) >= 2 and segs[1].isdigit():
        path = f"/{segs[0]}/{segs[1]}"
    elif segs:
        path = f"/{segs[0]}"
    else:
        path = "/"
    return f"{systems.system_for_url(url) or u.netloc}:{path}"
