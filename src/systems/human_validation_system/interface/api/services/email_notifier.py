"""
Email Notifier - Avisa o administrador por email (SMTP) quando uma rodada fecha
"""
import smtplib
import threading
from email.message import EmailMessage
from typing import Optional

from loguru import logger


class EmailNotifier:
    """
    Envia avisos por email sem travar a interface.

    Responsabilidades:
    - Enviar em segundo plano (thread) via SMTP com STARTTLS (ex.: Gmail + senha de app)
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
    ):
        self.recipient = recipient
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        if not self.enabled:
            logger.warning("Aviso por email desativado (configure HV_NOTIFY_EMAIL, HV_SMTP_USER e HV_SMTP_PASSWORD)")
        logger.debug(f"EmailNotifier inicializado (ativo={self.enabled})")

    @property
    def enabled(self) -> bool:
        return bool(self.recipient and self.user and self.password)

    def _deliver(self, message: EmailMessage) -> None:
        try:
            with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(self.user, self.password)
                smtp.send_message(message)
            logger.success(f"Email enviado para {self.recipient}: {message['Subject']}")
        except Exception as e:
            logger.error(f"Falha ao enviar email ({message['Subject']}): {e}")

    def send(self, subject: str, body: str) -> None:
        if not self.enabled:
            logger.info(f"[email desativado] {subject}")
            return
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.user
        message["To"] = self.recipient
        message.set_content(body)
        threading.Thread(target=self._deliver, args=(message,), daemon=True).start()
