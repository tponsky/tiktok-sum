"""
Email service using Resend
Reusable across multiple apps
"""
import os
import resend
from typing import Optional

# Initialize Resend
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY

# Default sender - use your verified domain
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@staycurrentapp.com")
APP_NAME = os.getenv("APP_NAME", "VidRecall")


def is_configured() -> bool:
    """Check if email service is configured."""
    return bool(RESEND_API_KEY)


def send_email(
    to: str,
    subject: str,
    html: str,
    from_email: str = None,
    reply_to: str = None
) -> Optional[dict]:
    """
    Send an email using Resend.

    Returns the response dict on success, None on failure.
    """
    if not is_configured():
        print("Email service not configured - RESEND_API_KEY not set")
        return None

    try:
        params = {
            "from": from_email or FROM_EMAIL,
            "to": [to],
            "subject": subject,
            "html": html,
        }
        if reply_to:
            params["reply_to"] = reply_to

        response = resend.Emails.send(params)
        print(f"Email sent to {to}: {response}")
        return response
    except Exception as e:
        print(f"Failed to send email to {to}: {e}")
        return None


def send_password_reset_email(to: str, reset_url: str, app_name: str = None) -> bool:
    """
    Send a password reset email.

    Returns True on success, False on failure.
    """
    app = app_name or APP_NAME

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 30px;
            }}
            .button {{
                display: inline-block;
                background: #fe2c55;
                color: white !important;
                text-decoration: none;
                padding: 12px 30px;
                border-radius: 6px;
                font-weight: 600;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 30px;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Reset Your Password</h2>
            <p>You requested to reset your password for your {app} account.</p>
            <p>Click the button below to set a new password:</p>
            <a href="{reset_url}" class="button">Reset Password</a>
            <p>This link will expire in 1 hour.</p>
            <p>If you didn't request this, you can safely ignore this email.</p>
            <div class="footer">
                <p>This email was sent by {app}</p>
            </div>
        </div>
    </body>
    </html>
    """

    result = send_email(
        to=to,
        subject=f"Reset your {app} password",
        html=html
    )

    return result is not None


def send_welcome_email(to: str, app_name: str = None) -> bool:
    """
    Send a welcome email to new users.

    Returns True on success, False on failure.
    """
    app = app_name or APP_NAME

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 30px;
            }}
            .highlight {{
                background: #e8f5e9;
                padding: 15px;
                border-radius: 6px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 30px;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Welcome to {app}!</h2>
            <p>Thanks for signing up. Your account has been created successfully.</p>
            <div class="highlight">
                <strong>You have $2.00 in free credits</strong> to get started!
            </div>
            <p>Start exploring by searching your video library or adding new videos.</p>
            <div class="footer">
                <p>This email was sent by {app}</p>
            </div>
        </div>
    </body>
    </html>
    """

    result = send_email(
        to=to,
        subject=f"Welcome to {app}!",
        html=html
    )

    return result is not None
