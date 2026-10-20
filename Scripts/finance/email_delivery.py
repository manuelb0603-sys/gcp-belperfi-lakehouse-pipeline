"""Weekly report email: the multipart message and its Gmail SMTP delivery."""

from __future__ import annotations

import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def report_subject(today: date | None = None) -> str:
    """Subject line of the weekly email (also the short-term memory subject)."""
    return f"Weekly Financial Progress Report Card {(today or date.today()).strftime('%Y-%m-%d')}"


def build_email_message(sender: str, receiver_header: str, report_text: str, plain_text: str | None = None) -> MIMEMultipart:
    """Wrap the HTML report in a multipart email so it works in both text and HTML clients."""
    msg_root = MIMEMultipart('related')
    msg_root['From'] = sender
    msg_root['To'] = receiver_header
    msg_root['Subject'] = report_subject()

    msg_alt = MIMEMultipart('alternative')
    msg_root.attach(msg_alt)

    if plain_text and plain_text.strip():
        # Agent mode supplies plain text derived from the validated report HTML.
        plain_body = plain_text.strip()
    else:
        plain_lines = [f"Weekly Financial Progress — {date.today().strftime('%B %d, %Y')}", ""]
        rpt_first = report_text.split('\n\n')[0] if report_text else ''
        if rpt_first:
            plain_lines.extend([rpt_first, ""])
        plain_body = "\n".join(plain_lines)
    msg_alt.attach(MIMEText(plain_body, 'plain'))
    msg_alt.attach(MIMEText(report_text, 'html'))

    return msg_root


def send_email(sender: str, receivers: list[str], password: str, msg_root: MIMEMultipart) -> None:
    """Deliver the final report via the configured Gmail account."""
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receivers, msg_root.as_string())
