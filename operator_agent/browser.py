"""A small stateful web browser for the operator.

It keeps cookies, follows redirects, and turns every page into (a) readable text and
(b) numbered links [L#] and forms [F#]. The operator can only submit forms that exist on the
page it is currently looking at, and hidden fields (CSRF tokens) are carried automatically,
so it behaves like a person using a browser rather than a script replaying requests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

MAX_TEXT_CHARS = 5000


class BrowserError(Exception):
    """Raised for problems the operator can act on (bad form id, missing field...)."""


@dataclass
class FormField:
    name: str
    type: str
    value: str = ""
    options: list[str] = field(default_factory=list)
    required: bool = False


@dataclass
class Form:
    id: str
    method: str
    action: str
    fields: list[FormField]

    @property
    def visible_fields(self) -> list[FormField]:
        return [f for f in self.fields if f.type != "hidden" and f.type != "submit"]


@dataclass
class Page:
    url: str
    status: int
    title: str
    text: str
    links: dict[str, tuple[str, str]]  # id -> (absolute href, text)
    forms: dict[str, Form]
    redirects: list[tuple[int, str]]
    is_login: bool = False
    alerts: list[str] = field(default_factory=list)

    def render(self) -> str:
        out = [f"URL: {self.url}  (HTTP {self.status})"]
        if self.redirects:
            out.append("Redirected via: " + " -> ".join(f"{s} {u}" for s, u in self.redirects))
        out.append(f"TITLE: {self.title}")
        out.append(self.text)
        if self.links:
            out.append("LINKS: " + "  ".join(f"[{i}] {t or h}" for i, (h, t) in self.links.items()))
        for f in self.forms.values():
            fields = ", ".join(
                f"{x.name}({'select: ' + '|'.join(x.options) if x.options else x.type}"
                f"{', required' if x.required else ''}{', default=' + x.value if x.value else ''})"
                for x in f.visible_fields)
            out.append(f"FORM [{f.id}] {f.method} {f.action} fields: {fields}  "
                       "(hidden fields such as CSRF tokens are added automatically)")
        return "\n".join(out)


def _squash(s: str) -> str:
    return " ".join(s.split())


class Browser:
    def __init__(self, timeout: float = 10.0):
        self.session = requests.Session()
        self.timeout = timeout
        self.page: Page | None = None

    # ----------------------------------------------------------- navigation
    def goto(self, url: str) -> Page:
        return self._request("GET", url, None)

    def click(self, link_id: str) -> Page:
        if self.page is None or link_id not in self.page.links:
            raise BrowserError(f"No link {link_id} on the current page. Current links: "
                               f"{list(self.page.links) if self.page else 'none (no page open)'}")
        return self.goto(self.page.links[link_id][0])

    def submit(self, form_id: str, values: dict[str, str]) -> Page:
        if self.page is None or form_id not in self.page.forms:
            raise BrowserError(f"No form {form_id} on the current page. Current forms: "
                               f"{list(self.page.forms) if self.page else 'none (no page open)'}. "
                               "Forms are only valid for the page you are currently on; reopen the page.")
        form = self.page.forms[form_id]
        known = {f.name: f for f in form.fields}
        unknown = [k for k in values if k not in known or known[k].type == "hidden"]
        if unknown:
            raise BrowserError(f"Unknown or non-editable field(s) {unknown}. Editable fields: "
                               f"{[f.name for f in form.visible_fields]}")
        data: dict[str, str] = {}
        for f in form.fields:
            if f.type == "submit":
                continue
            data[f.name] = str(values[f.name]) if f.name in values else f.value
        for f in form.visible_fields:
            if f.options and data[f.name] not in f.options:
                raise BrowserError(f"Field '{f.name}' must be one of {f.options}, got '{data[f.name]}'.")
        missing = [f.name for f in form.visible_fields if f.required and not data[f.name].strip()]
        if missing:
            raise BrowserError(f"Required field(s) left empty: {missing}.")
        return self._request(form.method, form.action, data)

    # ------------------------------------------------------------- internals
    def _request(self, method: str, url: str, data: dict | None) -> Page:
        try:
            kw = {"params": data} if method == "GET" else {"data": data}
            resp = self.session.request(method, url, timeout=self.timeout, allow_redirects=True, **kw)
        except requests.RequestException as exc:  # connection errors, timeouts
            raise ConnectionError(f"{type(exc).__name__}: {exc}") from exc
        self.page = self._parse(resp)
        return self.page

    def _parse(self, resp: requests.Response) -> Page:
        soup = BeautifulSoup(resp.text, "lxml")
        title = _squash(soup.title.get_text()) if soup.title else ""
        links: dict[str, tuple[str, str]] = {}
        forms: dict[str, Form] = {}
        alerts: list[str] = []

        for a in soup.find_all("a", href=True):
            lid = f"L{len(links) + 1}"
            links[lid] = (urljoin(resp.url, a["href"]), _squash(a.get_text()))
            a.replace_with(f"{_squash(a.get_text())} [{lid}]")

        for el in soup.select('[role="alert"]'):
            msg = _squash(el.get_text())
            alerts.append(msg)
            el.replace_with(soup.new_string(f"\n!! ALERT: {msg}\n"))
        for el in soup.select('[role="status"]'):
            el.replace_with(soup.new_string(f"\nNOTICE: {_squash(el.get_text())}\n"))

        for form in soup.find_all("form"):
            fid = f"F{len(forms) + 1}"
            fields: list[FormField] = []
            for inp in form.find_all(["input", "select", "textarea", "button"]):
                name = inp.get("name")
                if inp.name == "button" or inp.get("type") == "submit":
                    continue
                if not name:
                    continue
                if inp.name == "select":
                    opts = [o.get("value", o.get_text()) for o in inp.find_all("option")]
                    sel = inp.find("option", selected=True)
                    fields.append(FormField(name, "select", sel.get("value", "") if sel else opts[0] if opts else "",
                                            opts, inp.has_attr("required")))
                elif inp.name == "textarea":
                    fields.append(FormField(name, "textarea", inp.get_text(), [], inp.has_attr("required")))
                else:
                    fields.append(FormField(name, inp.get("type", "text"), inp.get("value", ""), [],
                                            inp.has_attr("required")))
            forms[fid] = Form(fid, (form.get("method") or "GET").upper(),
                              urljoin(resp.url, form.get("action") or resp.url), fields)
            form.replace_with(soup.new_string(f"\n[form {fid} here]\n"))

        for tr in soup.find_all("tr"):
            cells = [_squash(c.get_text()) for c in tr.find_all(["th", "td"])]
            tr.replace_with(soup.new_string("\n" + " | ".join(cells)))
        for br in soup.find_all(["h1", "h2", "h3", "p", "li", "div", "nav"]):
            br.insert_before("\n")
            br.insert_after("\n")
        body = soup.body.get_text() if soup.body else soup.get_text()
        lines = [_squash(l) for l in body.splitlines()]
        text = "\n".join(l for l in lines if l)
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + f"\n...[page truncated at {MAX_TEXT_CHARS} chars]"

        is_login = any(f.type == "password" for fm in forms.values() for f in fm.fields)
        redirects = [(r.status_code, r.headers.get("Location", "")) for r in resp.history]
        return Page(resp.url, resp.status_code, title, text, links, forms, redirects, is_login, alerts)
