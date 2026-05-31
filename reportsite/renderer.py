"""Static HTML site renderer for change reports.

Renders ``reports/_site/``:
  - ``_site/index.html`` — index of all rounds, newest first
  - ``_site/round-NNN/<component>.html`` — one page per report
  - ``_site/round-NNN/assets/*`` — copied asset files
  - ``_site/marked.min.js`` — vendored client-side markdown renderer

Markdown is rendered client-side via marked.js so the generator has no markdown
dependency.  Flame-graph assets (``*.html``) referenced in the body are wrapped
in ``<iframe>`` elements; other image assets are embedded as ``<img>``.

The palette mirrors the oversight deck (warm bone/amber/green/cyan on dark).
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from .model import Report, ReportIndex, RoundGroup

# ---------------------------------------------------------------------------
# Palette constants (CSS variables — defined once in the base style)
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg:      #1a1814;
  --surface: #221f1a;
  --border:  #3a352a;
  --bone:    #e9e1cf;
  --dim:     #b3a98f;
  --faint:   #7c735d;
  --green:   #76e0a0;
  --amber:   #f0b44b;
  --oxide:   #e0654f;
  --cyan:    #6fc6d6;
  --violet:  #b79be0;
}
*, *::before, *::after { box-sizing: border-box; }
html { font-size: 16px; }
body {
  background: var(--bg);
  color: var(--bone);
  font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
  margin: 0;
  padding: 0;
}
a { color: var(--cyan); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 980px; margin: 0 auto; padding: 2rem 1.5rem; }

/* masthead */
.masthead {
  border-bottom: 1px solid var(--border);
  padding: 1rem 0 0.75rem;
  margin-bottom: 2rem;
}
.masthead-title { font-size: 1.25rem; color: var(--bone); font-weight: bold; }
.masthead-sub   { font-size: 0.8rem; color: var(--faint); margin-top: 0.15rem; }

/* round group */
.round-header {
  border-left: 3px solid var(--amber);
  padding-left: 0.75rem;
  margin: 2.5rem 0 1rem;
}
.round-header h2 {
  margin: 0;
  font-size: 1rem;
  color: var(--amber);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.round-header .round-meta { font-size: 0.78rem; color: var(--faint); margin-top: 0.2rem; }

/* report card */
.report-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1rem 1.25rem;
  margin-bottom: 0.75rem;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
}
.report-card-left { flex: 1; min-width: 0; }
.report-card-component {
  font-size: 1rem;
  font-weight: bold;
  color: var(--bone);
}
.report-card-delta {
  font-size: 0.85rem;
  color: var(--green);
  margin-top: 0.2rem;
}
.report-card-meta {
  font-size: 0.75rem;
  color: var(--faint);
  margin-top: 0.15rem;
}
.verdict-accepted { color: var(--green); }
.verdict-rejected { color: var(--oxide); }
.verdict-pending  { color: var(--amber); }
.report-card-link {
  font-size: 0.8rem;
  color: var(--cyan);
  white-space: nowrap;
}

/* empty state */
.empty-state {
  text-align: center;
  padding: 6rem 2rem;
  color: var(--faint);
}
.empty-state h2 { color: var(--amber); font-size: 1.1rem; }
.empty-state p  { font-size: 0.85rem; margin-top: 0.5rem; }

/* report page */
.report-nav {
  font-size: 0.8rem;
  margin-bottom: 1.5rem;
  color: var(--faint);
}
.report-header {
  border-left: 3px solid var(--cyan);
  padding-left: 0.75rem;
  margin-bottom: 1.5rem;
}
.report-header h1 {
  margin: 0;
  font-size: 1.2rem;
  color: var(--bone);
}
.report-header-meta {
  font-size: 0.78rem;
  color: var(--faint);
  margin-top: 0.3rem;
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}
.report-header-delta { color: var(--green); font-size: 0.9rem; margin-top: 0.4rem; }

/* flamegraph embed */
.flamegraph-embed {
  width: 100%;
  height: 420px;
  border: 1px solid var(--border);
  border-radius: 4px;
  margin: 1rem 0;
  background: #111;
}

/* markdown body rendered by marked.js */
#report-body h1, #report-body h2 {
  color: var(--amber);
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.25rem;
  margin-top: 2rem;
}
#report-body h3 { color: var(--cyan); margin-top: 1.5rem; }
#report-body h4 { color: var(--dim); }
#report-body code {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 0.1em 0.35em;
  font-size: 0.88em;
}
#report-body pre {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1rem;
  overflow-x: auto;
  font-size: 0.82em;
  line-height: 1.55;
}
#report-body pre code { background: none; border: none; padding: 0; }
#report-body table {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.85em;
  margin: 1rem 0;
}
#report-body th {
  background: var(--surface);
  color: var(--dim);
  font-weight: normal;
  padding: 0.4rem 0.6rem;
  border: 1px solid var(--border);
  text-align: left;
}
#report-body td {
  padding: 0.35rem 0.6rem;
  border: 1px solid var(--border);
  vertical-align: top;
}
#report-body tr:nth-child(even) td { background: var(--surface); }
#report-body blockquote {
  border-left: 3px solid var(--amber);
  margin: 0;
  padding: 0.5rem 1rem;
  color: var(--dim);
  font-size: 0.9em;
}
#report-body img {
  max-width: 100%;
  border: 1px solid var(--border);
  border-radius: 4px;
}
"""

# Minimal marked.js — vendored (CDN fallback in the HTML as comment)
# We embed a lightweight shim that delegates to the CDN if the local copy is
# absent.  For offline use the generator writes assets/marked.min.js.
_MARKED_CDN = "https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _livereload_script(root_rel: str) -> str:
    """A dependency-free poller that reloads the page when the site is rebuilt.

    ``render_site`` writes ``_site/build-id.txt`` (a content hash) on every
    render; the served page fetches it once a second and reloads when the hash
    changes.  So a rebuilt site (e.g. after the ``post-commit`` hook fires on a
    new report) refreshes open tabs with no websocket/livereload dependency.
    """
    build_id_url = f"{root_rel}build-id.txt"
    return (
        "<script>\n"
        "(function () {\n"
        "  var current = null;\n"
        "  function poll() {\n"
        f"    fetch('{build_id_url}', {{cache: 'no-store'}})\n"
        "      .then(function (r) { return r.ok ? r.text() : Promise.reject(); })\n"
        "      .then(function (id) {\n"
        "        id = id.trim();\n"
        "        if (current === null) { current = id; return; }\n"
        "        if (id !== current) { location.reload(); }\n"
        "      })\n"
        "      .catch(function () {});\n"
        "  }\n"
        "  setInterval(poll, 1000);\n"
        "  poll();\n"
        "})();\n"
        "</script>"
    )


def _base_html(title: str, body_html: str, *, root_rel: str = "") -> str:
    """Wrap ``body_html`` in a full HTML document with inline CSS."""
    marked_src = f"{root_rel}marked.min.js" if root_rel else "marked.min.js"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html_escape(title)} · bt change reports</title>
<style>{_CSS}</style>
</head>
<body>
{body_html}
<script src="{marked_src}"></script>
{_livereload_script(root_rel)}
</body>
</html>"""


def _masthead(subtitle: str = "") -> str:
    sub = f'<div class="masthead-sub">{_html_escape(subtitle)}</div>' if subtitle else ""
    return (
        '<div class="masthead">'
        '<div class="wrap">'
        '<div class="masthead-title">bt · change reports</div>'
        f"{sub}"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


_VERDICT_CLASS = {
    "accepted": "verdict-accepted",
    "rejected": "verdict-rejected",
    "pending": "verdict-pending",
}
_VERDICT_GLYPH = {"accepted": "✓", "rejected": "✗", "pending": "◷"}


def _verdict_span(verdict: str) -> str:
    cls = _VERDICT_CLASS.get(verdict, "verdict-rejected")
    glyph = _VERDICT_GLYPH.get(verdict, "•")
    return f'<span class="{cls}">{glyph} {_html_escape(verdict)}</span>'


def _report_card(report: Report, report_href: str) -> str:
    fm = report.frontmatter
    return (
        f'<div class="report-card">'
        f'<div class="report-card-left">'
        f'<div class="report-card-component">{_html_escape(fm.component)}</div>'
        f'<div class="report-card-delta">{_html_escape(fm.headline_delta)}</div>'
        f'<div class="report-card-meta">'
        f"{_html_escape(fm.metric)} &nbsp;·&nbsp; "
        f"{_html_escape(fm.date)} &nbsp;·&nbsp; "
        f"PR #{fm.pr} &nbsp;·&nbsp; "
        f"{_verdict_span(fm.verdict)}"
        f"</div>"
        f"</div>"
        f'<a class="report-card-link" href="{report_href}">view report →</a>'
        f"</div>"
    )


def _round_section(group: RoundGroup) -> str:
    n = len(group.reports)
    noun = "change" if n == 1 else "changes"
    header = (
        f'<div class="round-header">'
        f"<h2>round {group.round:03d}</h2>"
        f'<div class="round-meta">{n} {noun}</div>'
        f"</div>"
    )
    cards = ""
    for report in group.reports:
        report_dir = f"round-{group.round:03d}"
        stem = report.source_path.stem
        href = f"{report_dir}/{stem}.html"
        cards += _report_card(report, href)
    return header + cards


def render_index(index: ReportIndex) -> str:
    """Render the index page HTML."""
    if index.is_empty:
        body = (
            _masthead("no reports yet")
            + '<div class="wrap">'
            + '<div class="empty-state">'
            + "<h2>Awaiting the first swarm</h2>"
            + "<p>Change reports appear here after a round of the improvement loop lands.</p>"
            + "<p>See <code>reports/README.md</code> for the report format and "
            + "<code>.claude/skills/change-report/SKILL.md</code> for the generator protocol.</p>"
            + "</div>"
            + "</div>"
        )
    else:
        rounds_html = "".join(_round_section(g) for g in index.rounds)
        body = (
            _masthead(f"{index.total_reports} reports across {len(index.rounds)} rounds")
            + '<div class="wrap">'
            + rounds_html
            + "</div>"
        )
    return _base_html("index", body)


# ---------------------------------------------------------------------------
# Report page
# ---------------------------------------------------------------------------

# Pattern: an image tag whose src ends with .html (a flamegraph HTML asset).
_IMG_HTML_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+\.html)\)")
# Pattern: an image tag whose src ends with .svg.
_IMG_SVG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+\.svg)\)")


def _preprocess_markdown(body: str, assets_rel: str) -> str:
    """Rewrite asset references in the markdown body before marked.js renders it.

    - ``![alt](assets/foo.html)`` → an ``<iframe>`` element (flame graphs).
    - ``![alt](assets/foo.svg)`` → an ``<img>`` element (keeps the native size).

    We replace the markdown image syntax with raw HTML that marked.js will pass
    through unchanged (marked treats raw HTML blocks as pass-through by default).
    """

    def html_embed(m: re.Match[str]) -> str:
        alt = _html_escape(m.group(1))
        src = m.group(2)
        # Rewrite relative assets/ path to be relative to the report page.
        # The markdown says ``assets/foo.html``; the page is at
        # ``_site/round-NNN/component.html`` so assets are at
        # ``_site/round-NNN/assets/foo.html`` — same relative path works.
        return (
            f'\n<iframe class="flamegraph-embed" src="{src}" '
            f'title="{alt}" loading="lazy"></iframe>\n'
        )

    def svg_embed(m: re.Match[str]) -> str:
        alt = _html_escape(m.group(1))
        src = m.group(2)
        return f'\n<img src="{src}" alt="{alt}">\n'

    body = _IMG_HTML_RE.sub(html_embed, body)
    body = _IMG_SVG_RE.sub(svg_embed, body)
    _ = assets_rel  # unused; kept for symmetry if callers pass it
    return body


def _section_headers_html(component: str, round_num: int) -> str:
    """Render the report header block (above the markdown body)."""
    return (
        f'<div class="report-header">'
        f"<h1>{_html_escape(component)} · round {round_num:03d}</h1>"
        f"</div>"
    )


def render_report_page(report: Report) -> str:
    """Render a single report page HTML (uses marked.js client-side)."""
    fm = report.frontmatter
    nav = (
        '<div class="wrap"><div class="report-nav"><a href="../index.html">← all reports</a></div>'
    )
    header = (
        '<div class="report-header">'
        f"<h1>{_html_escape(fm.component)} · round {fm.round:03d}</h1>"
        f'<div class="report-header-meta">'
        f"<span>{_html_escape(fm.date)}</span>"
        f"<span>PR #{fm.pr}</span>"
        f"<span>{_html_escape(fm.metric)}</span>"
        f"<span>{_verdict_span(fm.verdict)}</span>"
        f"</div>"
        f'<div class="report-header-delta">{_html_escape(fm.headline_delta)}</div>'
        "</div>"
    )

    preprocessed = _preprocess_markdown(report.body, assets_rel="assets/")
    # Embed the markdown as a <script> tag and render it into #report-body via
    # marked.js.  This avoids any server-side markdown library.
    escaped_md = preprocessed.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    script = (
        '<div id="report-body"></div>'
        "<script>"
        "document.addEventListener('DOMContentLoaded', function() {"
        f"  const md = `{escaped_md}`;"
        "  document.getElementById('report-body').innerHTML = marked.parse(md);"
        "});"
        "</script>"
    )

    body = _masthead() + nav + header + script + "</div>"
    title = f"{fm.component} · round {fm.round:03d}"
    return _base_html(title, body, root_rel="../")


# ---------------------------------------------------------------------------
# Site builder
# ---------------------------------------------------------------------------

# marked.min.js — fetched once from CDN and vendored; we ship a minimal shim
# that loads marked from CDN if the local copy doesn't define it.  The actual
# minified marked.js is downloaded at build time if curl/wget is available;
# otherwise this shim falls back to the CDN.
_MARKED_SHIM = f"""\
// marked.min.js shim — vendored by reportsite at build time.
// Falls back to CDN if the local copy is absent or empty.
if (typeof marked === 'undefined') {{
  var s = document.createElement('script');
  s.src = '{_MARKED_CDN}';
  document.head.appendChild(s);
}}
"""


def _fetch_marked_js(dest: Path) -> None:
    """Try to fetch marked.min.js from CDN; write shim on failure."""
    import subprocess
    import urllib.request

    try:
        with urllib.request.urlopen(_MARKED_CDN, timeout=5) as resp:
            dest.write_bytes(resp.read())
            return
    except Exception:
        pass

    # Try curl as a fallback.
    try:
        result = subprocess.run(
            ["curl", "-fsSL", "--max-time", "5", _MARKED_CDN],
            capture_output=True,
            check=True,
        )
        dest.write_bytes(result.stdout)
        return
    except Exception:
        pass

    # Neither worked — write the shim so pages still work via CDN.
    dest.write_text(_MARKED_SHIM, encoding="utf-8")


def render_site(reports_dir: Path, index: ReportIndex) -> None:
    """Render the full static site to ``reports_dir/_site/``.

    Layout::

        _site/
          index.html
          marked.min.js
          round-NNN/
            <component>.html
            assets/       ← copied from reports/round-NNN/assets/
    """
    site_dir = reports_dir / "_site"
    site_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate a content hash over every rendered page; written to
    # build-id.txt so the served page's poller (see _livereload_script) can
    # detect a rebuild and reload open tabs.  The pages embed a constant poll
    # script, never the hash itself, so there is no circularity.
    content_hash = hashlib.sha256()

    # index.html
    index_html = render_index(index)
    (site_dir / "index.html").write_text(index_html, encoding="utf-8")
    content_hash.update(index_html.encode("utf-8"))

    # marked.min.js
    marked_path = site_dir / "marked.min.js"
    if not marked_path.exists():
        _fetch_marked_js(marked_path)

    # Per-round directories + report pages + assets.
    for group in index.rounds:
        round_dir_name = f"round-{group.round:03d}"
        round_site_dir = site_dir / round_dir_name
        round_site_dir.mkdir(exist_ok=True)

        for report in group.reports:
            # Write the report HTML page.
            stem = report.source_path.stem
            page_html = render_report_page(report)
            (round_site_dir / f"{stem}.html").write_text(page_html, encoding="utf-8")
            content_hash.update(page_html.encode("utf-8"))

            # Copy assets directory.
            src_assets = report.source_path.parent / "assets"
            dst_assets = round_site_dir / "assets"
            if src_assets.is_dir():
                if dst_assets.exists():
                    shutil.rmtree(dst_assets)
                shutil.copytree(src_assets, dst_assets)

    # build-id.txt — bumped whenever any rendered page changes; the served page
    # polls it (once a second) and reloads when it differs.
    (site_dir / "build-id.txt").write_text(content_hash.hexdigest()[:12] + "\n", encoding="utf-8")
