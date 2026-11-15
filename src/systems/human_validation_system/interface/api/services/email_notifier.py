"""
Email Notifier - Avisa o administrador por email (Resend via HTTPS ou SMTP) quando uma rodada fecha
"""
import json
import smtplib
import threading
import urllib.request
from email.message import EmailMessage
from typing import Optional

from loguru import logger

from src.config.human_validation import RESEND_API_URL, RESEND_DEFAULT_SENDER


class EmailNotifier:
    """
    Envia avisos por email sem travar a interface.

    Responsabilidades:
    - Enviar em segundo plano (thread) pela API HTTPS do Resend, se houver chave (o Railway Hobby bloqueia SMTP)
    - Sem chave do Resend, enviar via SMTP com STARTTLS (ex.: Gmail + senha de app, uso local)
    - Ficar desativado, só registrando no log, se faltar destinatário ou credenciais
    - Nunca derrubar o fechamento da rodada: falhas de envio vão para o log
    """

    def __init__(
        self,
        recipient: Optional[str],
        host: str,
        port: int,
        user: Optional[str],
        password: Optional[str],
        resend_api_key: Optional[str] = None,
        resend_sender: str = RESEND_DEFAULT_SENDER,
    ):
        self.recipient = recipient
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.resend_api_key = resend_api_key
        self.resend_sender = resend_sender
        if not self.enabled:
            logger.warning(
                "Aviso por email desativado (configure HV_NOTIFY_EMAIL e HV_RESEND_API_KEY ou HV_SMTP_USER/HV_SMTP_PASSWORD)"
            )
        logger.debug(f"EmailNotifier inicializado (ativo={self.enabled}, via={self.channel})")

    @property
    def channel(self) -> str:
        return "resend" if self.resend_api_key else "smtp"

    @property
    def enabled(self) -> bool:
        return bool(self.recipient and (self.resend_api_key or (self.user and self.password)))

    def _deliver_resend(self, subject: str, body: str) -> None:
        payload = {"from": self.resend_sender, "to": [self.recipient], "subject": subject, "text": body}
        request = urllib.request.Request(
            RESEND_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.resend_api_key}",
                "Content-Type": "application/json",
                # O User-Agent padrão do urllib é barrado pelo Cloudflare do Resend
                "User-Agent": "llm-annotation-validacao-humana/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30):
            pass

    def _deliver_smtp(self, subject: str, body: str) -> None:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.user
        message["To"] = self.recipient
        message.set_content(body)
        with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(self.user, self.password)
            smtp.send_message(message)

    def _deliver(self, subject: str, body: str) -> None:
        try:
            if self.resend_api_key:
                self._deliver_resend(subject, body)
            else:
                self._deliver_smtp(subject, body)
            logger.success(f"Email enviado para {self.recipient} ({self.channel}): {subject}")
        except Exception as e:
            logger.error(f"Falha ao enviar email via {self.channel} ({subject}): {e}")

    def send(self, subject: str, body: str) -> None:
        if not self.enabled:
            logger.info(f"[email desativado] {subject}")
            return
        threading.Thread(target=self._deliver, args=(subject, body), daemon=True).start()
