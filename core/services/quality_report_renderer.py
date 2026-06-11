"""HTML/JSON rendering helpers for quality report envelopes."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from backend.core.contracts import QualityReport


def render_quality_report_html(report: QualityReport, *, title: str = "Lulynx Quality Report") -> str:
    """Render a self-contained human-readable HTML report.

    The embedded JSON remains the machine-readable source of truth. The HTML is
    only a presentation layer for manual review and release notes.
    """

    report_json = report.model_dump(mode="json")
    embedded_json = json.dumps(report_json, ensure_ascii=False, indent=2).replace("<", "\\u003c")
    rows = [
        _kv_row("Schema", report.schema_id),
        _kv_row("Artifact", report.artifact_path or "-"),
        _kv_row("Artifact kind", report.artifact_kind),
        _kv_row("Quality status", report.quality_status),
        _kv_row("Quality boundary", report.quality_boundary),
        _kv_row("Review level", report.review_level),
        _kv_row("Manual review", report.manual_review_status),
        _kv_row("Created", str(report.created_at)),
    ]
    sample_rows = "".join(
        f"<tr><td>{escape(sample.role)}</td><td>{escape(sample.path)}</td><td>{'yes' if sample.exists else 'no'}</td></tr>"
        for sample in report.samples
    ) or "<tr><td colspan=\"3\">No samples recorded.</td></tr>"
    metric_rows = "".join(
        "<tr>"
        f"<td>{escape(metric.name)}</td>"
        f"<td>{escape(str(metric.value))}</td>"
        f"<td>{escape(metric.status)}</td>"
        f"<td>{escape(metric.level)}</td>"
        f"<td>{escape(metric.source)}</td>"
        "</tr>"
        for metric in report.metrics
    ) or "<tr><td colspan=\"5\">No optional metrics recorded.</td></tr>"
    issue_rows = "".join(
        "<tr>"
        f"<td>{escape(str(issue.get('severity', '')))}</td>"
        f"<td>{escape(str(issue.get('code', '')))}</td>"
        f"<td>{escape(str(issue.get('message', '')))}</td>"
        "</tr>"
        for issue in report.issues
    ) or "<tr><td colspan=\"3\">No issues recorded.</td></tr>"
    evidence_summary = json.dumps(report.evidence_summary, ensure_ascii=False, indent=2)
    metadata = json.dumps(report.report_metadata, ensure_ascii=False, indent=2)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: Inter, Segoe UI, Arial, sans-serif; }}
    body {{ margin: 0; background: #101418; color: #eef3f8; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 32px 24px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: 720; }}
    h2 {{ margin: 28px 0 12px; font-size: 17px; }}
    p {{ color: #aab8c5; margin: 0 0 20px; }}
    table {{ width: 100%; border-collapse: collapse; border: 1px solid #29343f; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #29343f; vertical-align: top; }}
    th {{ background: #18212a; color: #dbe8f2; font-size: 13px; }}
    td {{ color: #eef3f8; font-size: 13px; }}
    .status {{ display: inline-block; padding: 4px 8px; border-radius: 6px; background: #253449; color: #d7e9ff; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #141c24; border: 1px solid #29343f; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
<main>
  <h1>{escape(title)}</h1>
  <p>Human-readable view of a request-native quality evidence envelope. A smoke pass is not a final quality pass.</p>
  <h2>Summary</h2>
  <table><tbody>{''.join(rows)}</tbody></table>
  <h2>Samples</h2>
  <table><thead><tr><th>Role</th><th>Path</th><th>Exists</th></tr></thead><tbody>{sample_rows}</tbody></table>
  <h2>Metrics</h2>
  <table><thead><tr><th>Name</th><th>Value</th><th>Status</th><th>Level</th><th>Source</th></tr></thead><tbody>{metric_rows}</tbody></table>
  <h2>Issues</h2>
  <table><thead><tr><th>Severity</th><th>Code</th><th>Message</th></tr></thead><tbody>{issue_rows}</tbody></table>
  <h2>Evidence Summary</h2>
  <pre>{escape(evidence_summary)}</pre>
  <h2>Metadata</h2>
  <pre>{escape(metadata)}</pre>
  <script id=\"quality-report-json\" type=\"application/json\">{embedded_json}</script>
</main>
</body>
</html>
"""


def write_quality_report_exports(
    report: QualityReport,
    output_dir: Path,
    *,
    basename: str = "quality_report",
) -> dict[str, str]:
    """Write JSON and HTML report exports and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_basename = _safe_basename(basename)
    json_path = output_dir / f"{safe_basename}.json"
    html_path = output_dir / f"{safe_basename}.html"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    html_path.write_text(render_quality_report_html(report), encoding="utf-8")
    return {"json_path": str(json_path), "html_path": str(html_path)}


def _kv_row(key: str, value: Any) -> str:
    text = str(value)
    if key == "Quality status":
        text = f'<span class="status">{escape(text)}</span>'
        return f"<tr><th>{escape(key)}</th><td>{text}</td></tr>"
    return f"<tr><th>{escape(key)}</th><td>{escape(text)}</td></tr>"


def _safe_basename(value: str) -> str:
    text = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "").strip())
    return text.strip("._") or "quality_report"
