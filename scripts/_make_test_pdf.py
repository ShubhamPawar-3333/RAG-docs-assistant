"""
Generate data/test_corpus/07_nimbus_onboarding.pdf without any third-party PDF library.

The repo has no reportlab/fpdf, but the test corpus needs a real PDF so the
PyPDFLoader ingestion path is exercised. This writes a minimal but valid
single-stream PDF with one text line per source line, which pypdf/PyPDFLoader
extract cleanly.
"""

from pathlib import Path

LINES = [
    "Nimbus Analytics - Onboarding Guide",
    "",
    "Every new workspace starts with a 14-day free trial of the Pro plan.",
    "No credit card is required to start the trial.",
    "",
    "Setup steps:",
    "Step 1: Create your workspace and name your first project.",
    "Step 2: Install an SDK or paste the tracking snippet into your page <head>.",
    "Step 3: Open Live View and confirm your first events are arriving.",
    "Step 4: Invite your teammates from Workspace settings > Members.",
    "Step 5: Build your first dashboard from a template.",
    "",
    "Default settings for a new workspace:",
    "- Timezone: UTC",
    "- Session timeout: 30 minutes",
    "- Default plan after trial: Free (unless you add a payment method)",
    "",
    "Onboarding support:",
    "Email onboarding@nimbusanalytics.com or join office hours every Tuesday.",
    "A dedicated onboarding manager is assigned to Enterprise customers only.",
]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(lines):
    # Content stream: start at top, move down 16pt per line.
    parts = ["BT", "/F1 12 Tf", "16 TL", "72 720 Td"]
    for line in lines:
        parts.append(f"({_escape(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    content = "\n".join(parts).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
        + f"startxref\n{xref_pos}\n%%EOF".encode()
    )
    return bytes(out)


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "data" / "test_corpus" / "07_nimbus_onboarding.pdf"
    target.write_bytes(build_pdf(LINES))
    print(f"wrote {target} ({target.stat().st_size} bytes)")
