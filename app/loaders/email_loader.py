"""
Email Loader — stdlib email: parse .eml/.msg; headers (From, To, Subject, Date) as metadata.
"""
from pathlib import Path
import email
from email import policy

from langchain_core.documents import Document

from app.loaders.base import BaseLoader


class EmailLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".eml", ".msg"]

    def load(self, file_path: Path) -> list[Document]:
        raw = file_path.read_bytes()
        msg = email.message_from_bytes(raw, policy=policy.default)
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode(errors="replace")
                    break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(errors="replace")
        metadata = {
            "source": file_path.name,
            "subject": msg.get("Subject", ""),
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "date": msg.get("Date", ""),
        }
        return [Document(page_content=body or "(no body)", metadata=metadata)]
