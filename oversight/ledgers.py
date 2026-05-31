"""Parse the two append-only ledgers into structured entries.

``IMPROVEMENTS.md`` — one block per round (newest at the bottom):

    ## <date> — <component>: <one-line target>  [accepted | rejected]
    metric:   <name> — <before> → <after> (<delta>)
    eval:     golden unchanged within <tol>  |  accuracy moved: ...
    PR:       <url or #number>
    note:     <why accepted, or which gate rejected it and the takeaway>

``API_REQUESTS.md`` — one block per request:

    ## <date> — <requester> needs <field/data> from <producer>
    why:    <one line>
    status: open | accepted | done

Both files ship as empty templates (only the header + the fenced format spec).
The parsers skip the fenced ```` ``` ```` example block and tolerate the
empty-template state by returning empty lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ``## YYYY-MM-DD — rest`` ; the em-dash separates date from the body.
_HEADER = re.compile(r"^##\s+(?P<date>\S+)\s+—\s+(?P<rest>.+?)\s*$")
_VERDICT = re.compile(r"\[(accepted|rejected|pending[^\]]*)\]\s*$", re.IGNORECASE)
_IMPROVE_HEAD = re.compile(r"^(?P<component>[^:]+):\s*(?P<target>.+)$")
_REQUEST_HEAD = re.compile(
    r"^(?P<requester>.+?)\s+needs\s+(?P<field>.+?)\s+from\s+(?P<producer>.+)$"
)


@dataclass(frozen=True)
class ImprovementEntry:
    date: str
    component: str
    target: str
    verdict: str  # accepted | rejected | pending | unknown
    metric: str
    eval_note: str
    pr: str
    note: str


@dataclass(frozen=True)
class RequestEntry:
    date: str
    requester: str
    field: str
    producer: str
    why: str
    status: str  # open | accepted | done | unknown


def _blocks(text: str) -> list[tuple[str, list[str]]]:
    """Split the ledger into (header-line, body-lines) blocks, skipping any
    fenced ```` ``` ```` example region (the format spec at the top)."""
    blocks: list[tuple[str, list[str]]] = []
    header: str | None = None
    body: list[str] = []
    in_fence = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("## "):
            if header is not None:
                blocks.append((header, body))
            header = stripped
            body = []
        elif header is not None:
            body.append(stripped)

    if header is not None:
        blocks.append((header, body))
    return blocks


def _field(body: list[str], key: str) -> str:
    prefix = f"{key}:"
    for line in body:
        if line.lower().startswith(prefix.lower()):
            return line[len(prefix) :].strip()
    return ""


def parse_improvements(text: str) -> list[ImprovementEntry]:
    entries: list[ImprovementEntry] = []
    for header, body in _blocks(text):
        m = _HEADER.match(header)
        if m is None:
            continue
        rest = m.group("rest")

        verdict = "unknown"
        vm = _VERDICT.search(rest)
        if vm is not None:
            verdict = vm.group(1).lower().split()[0]
            rest = rest[: vm.start()].strip()

        component = rest
        target = ""
        hm = _IMPROVE_HEAD.match(rest)
        if hm is not None:
            component = hm.group("component").strip()
            target = hm.group("target").strip()

        entries.append(
            ImprovementEntry(
                date=m.group("date"),
                component=component,
                target=target,
                verdict=verdict,
                metric=_field(body, "metric"),
                eval_note=_field(body, "eval"),
                pr=_field(body, "PR"),
                note=_field(body, "note"),
            )
        )
    return entries


def parse_requests(text: str) -> list[RequestEntry]:
    entries: list[RequestEntry] = []
    for header, body in _blocks(text):
        m = _HEADER.match(header)
        if m is None:
            continue
        rest = m.group("rest")

        requester = rest
        field = ""
        producer = ""
        rm = _REQUEST_HEAD.match(rest)
        if rm is not None:
            requester = rm.group("requester").strip()
            field = rm.group("field").strip()
            producer = rm.group("producer").strip()

        status = _field(body, "status").lower().split()[0] if _field(body, "status") else "unknown"

        entries.append(
            RequestEntry(
                date=m.group("date"),
                requester=requester,
                field=field,
                producer=producer,
                why=_field(body, "why"),
                status=status or "unknown",
            )
        )
    return entries


def read_improvements(path: Path) -> list[ImprovementEntry]:
    if not path.exists():
        return []
    return parse_improvements(path.read_text())


def read_requests(path: Path) -> list[RequestEntry]:
    if not path.exists():
        return []
    return parse_requests(path.read_text())


__all__ = [
    "ImprovementEntry",
    "RequestEntry",
    "parse_improvements",
    "parse_requests",
    "read_improvements",
    "read_requests",
]
