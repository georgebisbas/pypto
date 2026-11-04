# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Render a step-by-step HTML descent of one kernel's compilation.

The page groups stages by representation (PyPTO IR, PTO MLIR, cce-mlir's own
passes, device LLVM) and shows each stage as a line diff against the previous
stage in the same group. Diffs never cross a representation boundary: the page
marks that boundary instead, because a diff between two languages is noise.

``pypto.tools.ir_trace`` renders PyPTO pass dumps alone with Python-aware
function sections. This module extends the story past Buffer IR to the MLIR and
LLVM stages and uses a representation-agnostic line diff.
"""

from __future__ import annotations

import difflib
import html
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Stage", "collect_stages", "render_trace_html", "write_trace_html"]

_CONTEXT_LINES = 3

# Layout: a sticky stage index beside one column of stage cards; single column
# under 960px. Palette: cool slate neutrals with a single indigo accent and
# semantic green/red reserved for diff lines.
_STYLE = """
:root {
  --bg: #f2f5fa;
  --surface: #ffffff;
  --surface-alt: #e9eef6;
  --text: #161a22;
  --muted: #5a6474;
  --border: #cbd3e1;
  --accent: #2f43c6;
  --accent-soft: #e4e8fb;
  --add-bg: #ddf4e3;
  --add-fg: #11512c;
  --del-bg: #fde3e1;
  --del-fg: #7a1c19;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1218;
    --surface: #171b23;
    --surface-alt: #1f2530;
    --text: #e6e9f0;
    --muted: #9aa4b5;
    --border: #2c3442;
    --accent: #8fa2ff;
    --accent-soft: #232b45;
    --add-bg: #123a24;
    --add-fg: #a6e8bd;
    --del-bg: #43191a;
    --del-fg: #f3a8a2;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1218;
  --surface: #171b23;
  --surface-alt: #1f2530;
  --text: #e6e9f0;
  --muted: #9aa4b5;
  --border: #2c3442;
  --accent: #8fa2ff;
  --accent-soft: #232b45;
  --add-bg: #123a24;
  --add-fg: #a6e8bd;
  --del-bg: #43191a;
  --del-fg: #f3a8a2;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.5;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
}
.wrap { max-width: 1200px; margin-inline: auto; padding-block: 28px 64px; }
header.page { display: flex; flex-direction: column; gap: 10px; margin-bottom: 26px; }
h1 { font-size: 1.65rem; line-height: 1.2; margin: 0; text-wrap: balance; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 0.92rem; margin: 0; }
.sub code { font-family: var(--mono); font-size: 0.86rem; }
.controls {
  display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center;
  margin-top: 6px; padding: 10px 14px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px;
}
.controls label { display: inline-flex; gap: 7px; align-items: center; font-size: 0.9rem; cursor: pointer; }
.controls button {
  font: inherit; font-size: 0.86rem; padding: 4px 11px; border-radius: 6px; cursor: pointer;
  border: 1px solid var(--border); background: var(--surface-alt); color: var(--text);
}
.controls button:hover { border-color: var(--accent); }
.controls button:focus-visible, a:focus-visible, summary:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px;
}
.layout { display: grid; grid-template-columns: 260px minmax(0, 1fr); gap: 28px; align-items: start; }
@media (max-width: 960px) {
  .layout { grid-template-columns: minmax(0, 1fr); }
  nav.index { position: static; }
}
nav.index {
  position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow-y: auto;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px;
}
nav.index h2 {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
  margin: 14px 0 6px;
}
nav.index h2:first-child { margin-top: 0; }
nav.index ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
nav.index a {
  display: flex; gap: 8px; padding: 3px 6px; border-radius: 5px; text-decoration: none;
  color: var(--text); font-size: 0.85rem; font-family: var(--mono);
}
nav.index a:hover { background: var(--accent-soft); }
nav.index a .n { color: var(--muted); min-width: 2.2em; font-variant-numeric: tabular-nums; }
nav.index a.unchanged { color: var(--muted); }
main { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
section.group { display: flex; flex-direction: column; gap: 12px; }
.group-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; margin-top: 8px; }
.group-head h2 { font-size: 1.02rem; margin: 0; letter-spacing: 0.01em; }
.group-head .rep { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); }
.boundary {
  font-size: 0.82rem; color: var(--muted); border-left: 3px solid var(--accent);
  padding: 2px 0 2px 10px; margin: 4px 0;
}
.stage { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.stage > details > summary {
  cursor: pointer; list-style: none; display: flex; flex-wrap: wrap; align-items: baseline;
  gap: 4px 12px; padding: 10px 14px; user-select: none;
}
.stage > details > summary::-webkit-details-marker { display: none; }
.stage > details > summary:hover { background: var(--surface-alt); }
.stage .idx {
  font-family: var(--mono); font-size: 0.78rem; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.stage .name { font-family: var(--mono); font-size: 0.92rem; font-weight: 500; }
.stage .counts { margin-left: auto; display: flex; gap: 8px; font-family: var(--mono); font-size: 0.8rem; }
.stage .add-c { color: var(--add-fg); background: var(--add-bg); padding: 0 6px; border-radius: 4px; }
.stage .del-c { color: var(--del-fg); background: var(--del-bg); padding: 0 6px; border-radius: 4px; }
.stage .badge { color: var(--muted); font-size: 0.8rem; font-style: italic; }
.diff-scroll { overflow-x: auto; border-top: 1px solid var(--border); }
table.diff {
  border-collapse: collapse; width: max-content; min-width: 100%;
  font-family: var(--mono); font-size: 0.8rem; line-height: 1.45;
}
table.diff td { padding: 0 10px; vertical-align: top; white-space: pre; }
table.diff td.ln {
  text-align: right; color: var(--muted); background: var(--surface-alt);
  user-select: none; font-variant-numeric: tabular-nums; min-width: 3.2em;
}
table.diff td.sign { width: 1.4em; text-align: center; user-select: none; }
table.diff tr.add td { background: var(--add-bg); color: var(--add-fg); }
table.diff tr.del td { background: var(--del-bg); color: var(--del-fg); }
table.diff tr.skip td {
  color: var(--muted); background: var(--surface-alt); font-style: italic; text-align: center;
}
.full { border-top: 1px solid var(--border); }
pre.plain {
  margin: 0; padding: 12px 14px; overflow-x: auto; font-family: var(--mono);
  font-size: 0.8rem; line-height: 1.45; white-space: pre;
}
body.only-changed .stage.unchanged { display: none; }
body.only-changed nav.index a.unchanged { display: none; }
footer.page { margin-top: 34px; color: var(--muted); font-size: 0.8rem; }
"""

_SCRIPT = """
(function () {
  var filter = document.getElementById("only-changed");
  if (filter) {
    filter.addEventListener("change", function () {
      document.body.classList.toggle("only-changed", filter.checked);
    });
  }
  document.getElementById("expand-all")?.addEventListener("click", function () {
    document.querySelectorAll("article.stage details").forEach(function (d) { d.open = true; });
  });
  document.getElementById("collapse-all")?.addEventListener("click", function () {
    document.querySelectorAll("article.stage details").forEach(function (d) { d.open = false; });
  });
})();
"""


@dataclass(frozen=True)
class Stage:
    """One stage of the descent within a representation group."""

    group: str
    name: str
    text: str
    representation: str = ""


def _pass_file_key(path: Path) -> tuple[int, str]:
    """Order cce-mlir's ``NN_<pass>.mlir`` dumps numerically, then by name."""
    match = re.match(r"^(\d+)_", path.name)
    return (int(match.group(1)) if match else 1 << 30, str(path))


def _pass_name(path: Path) -> str:
    """Strip the numeric prefix from a cce-mlir dump file name."""
    return re.sub(r"^\d+_", "", path.stem)


def collect_stages(output: Path, compiler: str) -> list[Stage]:
    """Collect every stage of a traced kernel, grouped by representation.

    Args:
        output: Artifact directory written by ``trace_kernel``.
        compiler: ``cce_mlir`` or ``ptoas``.

    Returns:
        Stages in pipeline order, with group and representation labels.
    """
    from pypto.tools.ir_trace.discovery import discover_snapshots  # noqa: PLC0415

    stages: list[Stage] = []
    dumps = output / "passes_dump"
    if dumps.is_dir():
        for snapshot in discover_snapshots(dumps):
            stages.append(
                Stage(
                    group="PyPTO IR",
                    name=snapshot.pass_name or "frontend",
                    text=snapshot.text,
                    representation="PyPTO DSL",
                )
            )

    if compiler == "cce_mlir":
        emitted = sorted(
            path for path in (output / "cce_mlir").glob("*.mlir") if not path.name.endswith(".lowered.mlir")
        )
        for path in emitted:
            stages.append(
                Stage(
                    group="PTO MLIR",
                    name=path.stem,
                    text=path.read_text(encoding="utf-8"),
                    representation="PTO MLIR",
                )
            )
        passes = output / "cce_passes"
        if passes.is_dir():
            for path in sorted(passes.rglob("*.mlir"), key=_pass_file_key):
                stages.append(
                    Stage(
                        group="cce-mlir pipeline",
                        name=_pass_name(path),
                        text=path.read_text(encoding="utf-8"),
                        representation="cce / hivm MLIR",
                    )
                )
        for path in sorted((output / "cce_mlir").glob("*.lowered.mlir")):
            stages.append(
                Stage(
                    group="Device LLVM",
                    name=path.stem,
                    text=path.read_text(encoding="utf-8"),
                    representation="LLVM dialect",
                )
            )
    else:
        for path in sorted(output.rglob("*.pto")):
            stages.append(
                Stage(
                    group="PTO MLIR",
                    name=path.stem,
                    text=path.read_text(encoding="utf-8"),
                    representation="PTO MLIR",
                )
            )
        for path in sorted((output / "ptoas").glob("*.cpp")):
            stages.append(
                Stage(
                    group="ptoas output",
                    name=path.stem,
                    text=path.read_text(encoding="utf-8"),
                    representation="pto-isa C++",
                )
            )
    return stages


def _diff_rows(before: str, after: str) -> tuple[list[tuple[str, str, str, str]], int, int]:
    """Compute diff rows with collapsed context, plus inserted/deleted counts.

    Returns rows of ``(kind, before_number, after_number, text)`` where ``kind``
    is ``add``, ``del``, ``ctx`` or ``skip``.
    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    rows: list[tuple[str, str, str, str]] = []
    inserted = deleted = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            run = i2 - i1
            if run > 2 * _CONTEXT_LINES:
                head_end = _CONTEXT_LINES
                tail_start = run - _CONTEXT_LINES
            else:
                head_end = run
                tail_start = run
            for offset in range(head_end):
                rows.append(("ctx", str(i1 + offset + 1), str(j1 + offset + 1), before_lines[i1 + offset]))
            if tail_start > head_end:
                rows.append(("skip", "", "", f"⋯ {tail_start - head_end} unchanged lines"))
            for offset in range(tail_start, run):
                rows.append(("ctx", str(i1 + offset + 1), str(j1 + offset + 1), before_lines[i1 + offset]))
        elif tag == "delete":
            for offset in range(i1, i2):
                rows.append(("del", str(offset + 1), "", before_lines[offset]))
                deleted += 1
        elif tag == "insert":
            for offset in range(j1, j2):
                rows.append(("add", "", str(offset + 1), after_lines[offset]))
                inserted += 1
        else:  # replace: deletions first, then insertions
            for offset in range(i1, i2):
                rows.append(("del", str(offset + 1), "", before_lines[offset]))
                deleted += 1
            for offset in range(j1, j2):
                rows.append(("add", "", str(offset + 1), after_lines[offset]))
                inserted += 1
    return rows, inserted, deleted


def _render_diff(before: str, after: str) -> tuple[str, int, int]:
    rows, inserted, deleted = _diff_rows(before, after)
    parts = ['<div class="diff-scroll"><table class="diff"><tbody>']
    for kind, before_number, after_number, text in rows:
        escaped = html.escape(text, quote=False)
        if kind == "skip":
            parts.append(f'<tr class="skip"><td colspan="4">{escaped}</td></tr>')
            continue
        sign = {"add": "+", "del": "−", "ctx": " "}[kind]
        parts.append(
            f'<tr class="{kind}"><td class="ln">{before_number}</td>'
            f'<td class="ln">{after_number}</td>'
            f'<td class="sign">{sign}</td><td>{escaped}</td></tr>'
        )
    parts.append("</tbody></table></div>")
    return "".join(parts), inserted, deleted


def _render_stage(position: int, stage: Stage, previous: Stage | None) -> tuple[str, bool, int, int]:
    """Render one stage card, returning markup, changed flag and line counts."""
    if previous is None:
        # First stage of a representation has nothing to diff against.
        body = f'<div class="full"><pre class="plain">{html.escape(stage.text, quote=False)}</pre></div>'
        counts = ""
        changed, inserted, deleted = True, 0, 0
        open_attr = " open"
    else:
        body, inserted, deleted = _render_diff(previous.text, stage.text)
        changed = inserted > 0 or deleted > 0
        if changed:
            counts = (
                f'<span class="counts"><span class="add-c">+{inserted}</span>'
                f'<span class="del-c">−{deleted}</span></span>'
            )
            open_attr = " open"
        else:
            counts = '<span class="counts"><span class="badge">unchanged</span></span>'
            open_attr = ""

    css = "" if changed else " unchanged"
    head = (
        f'<summary><span class="idx">{position:02d}</span>'
        f'<span class="name">{html.escape(stage.name)}</span>{counts}</summary>'
    )
    markup = (
        f'<article class="stage{css}" id="stage-{position}">'
        f"<details{open_attr}>{head}{body}</details></article>"
    )
    return markup, changed, inserted, deleted


def render_trace_html(stages: list[Stage], title: str, subtitle: str) -> str:
    """Render the descent as a self-contained page fragment.

    The fragment is browser-openable as a file and publishable as an artifact:
    it carries its own ``<title>``, meta and styles but no ``<html>``/``<body>``.

    Args:
        stages: Ordered stages from :func:`collect_stages`.
        title: Page title (also the browser-tab name).
        subtitle: One sentence shown under the title.

    Returns:
        The HTML text.
    """
    # Stage numbers restart per representation group, and a diff is only taken
    # against the previous stage of the same group.
    numbered: list[tuple[int, Stage, Stage | None]] = []
    seen: dict[str, int] = {}
    previous: dict[str, Stage] = {}
    for stage in stages:
        seen[stage.group] = seen.get(stage.group, 0) + 1
        numbered.append((seen[stage.group], stage, previous.get(stage.group)))
        previous[stage.group] = stage

    entries = []
    for position, (index, stage, earlier) in enumerate(numbered, start=1):
        markup, changed, inserted, deleted = _render_stage(position, stage, earlier)
        entries.append(
            {
                "position": position,
                "index": index,
                "stage": stage,
                "markup": markup,
                "changed": changed,
                "inserted": inserted,
                "deleted": deleted,
            }
        )

    groups: list[str] = []
    for entry in entries:
        if not groups or groups[-1] != entry["stage"].group:
            groups.append(entry["stage"].group)

    nav = ['<nav class="index" aria-label="Stages">']
    for group in groups:
        nav.append(f"<h2>{html.escape(group)}</h2><ol>")
        for entry in entries:
            if entry["stage"].group != group:
                continue
            css = "" if entry["changed"] else ' class="unchanged"'
            nav.append(
                f'<li><a href="#stage-{entry["position"]}"{css}>'
                f'<span class="n">{entry["index"]:02d}</span>'
                f"{html.escape(entry['stage'].name)}</a></li>"
            )
        nav.append("</ol>")
    nav.append("</nav>")

    main: list[str] = []
    for group in groups:
        first = next(entry for entry in entries if entry["stage"].group == group)
        if main:
            main.append(
                '<p class="boundary">Representation changes here — diffs restart in '
                f"{html.escape(first['stage'].representation or group)}.</p>"
            )
        main.append(
            f'<section class="group" id="group-{first["position"]}">'
            f'<div class="group-head"><h2>{html.escape(group)}</h2>'
            f'<span class="rep">{html.escape(first["stage"].representation)}</span></div>'
        )
        main.extend(entry["markup"] for entry in entries if entry["stage"].group == group)
        main.append("</section>")

    changed_count = sum(1 for entry in entries if entry["changed"])
    return f"""<title>{html.escape(title)}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{_STYLE}</style>
<div class="wrap">
<header class="page">
  <h1>{html.escape(title)}</h1>
  <p class="sub">{subtitle}</p>
  <div class="controls">
    <label><input type="checkbox" id="only-changed"> Only stages that changed</label>
    <button type="button" id="expand-all">Expand all</button>
    <button type="button" id="collapse-all">Collapse all</button>
    <span class="sub">{len(stages)} stages · {changed_count} changed · {len(groups)} representations</span>
  </div>
</header>
<div class="layout">
{"".join(nav)}
<main>
{"".join(main)}
<footer class="page">Generated by <code>python -m pypto.tools.trace_kernel</code>. Each stage is diffed
against the previous stage in the same representation; ⋯ marks unchanged context.</footer>
</main>
</div>
</div>
<script>{_SCRIPT}</script>
"""


def write_trace_html(output: Path, compiler: str, name: str) -> Path:
    """Collect the stages under ``output`` and write ``trace.html`` beside them.

    Args:
        output: Artifact directory written by ``trace_kernel``.
        compiler: ``cce_mlir`` or ``ptoas``.
        name: Kernel name used in the page title.

    Returns:
        Path to the written page.
    """
    stages = collect_stages(output, compiler)
    subtitle = (
        f"{html.escape(compiler)} route · artifacts in <code>{html.escape(str(output))}</code>. "
        "Each stage is diffed against the previous stage in the same representation."
    )
    page = output / "trace.html"
    page.write_text(render_trace_html(stages, f"{name} IR descent", subtitle), encoding="utf-8")
    return page
