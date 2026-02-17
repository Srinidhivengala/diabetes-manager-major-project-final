# real_smtp.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# 🔐 CONFIG (keep safe, better use ENV in production)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SMTP_EMAIL = "srinidhivengala28@gmail.com"
SMTP_PASSWORD = "uimujurohbtgvvro"   # Gmail App Password (no spaces)


def send_otp_email(to_email: str, otp_code: str) -> bool:
    """
    Send OTP email using Gmail SMTP
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_EMAIL
        msg["To"] = to_email
        msg["Subject"] = "Your OTP Verification Code"

        body = f"""
Hello,

Your One-Time Password (OTP) is:

👉 {otp_code}

This OTP is valid for 10 minutes.
Do not share this code with anyone.

If you did not request this, please ignore this email.

Thanks,
Diabetes Manager Team
"""
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"[SMTP] ✅ OTP sent to {to_email}")
        return True

    except Exception as e:
        print("[SMTP] ❌ Failed to send OTP:", str(e))
        return False
