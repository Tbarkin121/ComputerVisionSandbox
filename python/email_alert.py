import os
from twilio.rest import Client

# Find these values at https://twilio.com/user/account
# To set up environmental variables, see http://twil.io/secure
account_sid = os.environ[' Replace with your Twilio Account SID']
auth_token = os.environ[' Replace with your Twilio Account SID']

client = Client(account_sid, auth_token)

client.api.account.messages.create(
    to="Replace with the recipient's phone number",
    from_="Replace with your Twilio phone number",
    body="Hello there!")


#%%

from twilio.rest import Client

# Twilio credentials
ACCOUNT_SID = " Replace with your Twilio Account SID"  # Replace with your Twilio Account SID
AUTH_TOKEN = " Replace with your Twilio Account SID"    # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = "Replace with your Twilio phone number"  # Replace with your Twilio phone number
RECIPIENT_PHONE_NUMBER = "Replace with the recipient's phone number"  # Replace with the recipient's phone number

def send_sms(message):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    try:
        sms = client.messages.create(
            body=message,  # The content of your SMS
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
        print(f"Message sent! SID: {sms.sid}")
    except Exception as e:
        print(f"Failed to send message: {e}")

# Example usage
if __name__ == "__main__":
    send_sms("Alert! A fall was detected!")
    
#%%

import smtplib
from email.mime.text import MIMEText

# Gmail credentials
EMAIL_ADDRESS = "Replace with your Gmail address"  # Replace with your Gmail address
EMAIL_PASSWORD = "Replace with your Gmail App Password"    # Replace with your Gmail App Password

# Recipient's email address (e.g., Verizon SMS gateway)
RECIPIENT_EMAIL = "Replace with the recipient's SMS gateway"  # Replace with the recipient's SMS gateway

def send_email(subject, body):
    # Set up the Gmail SMTP server
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    try:
        # Create the email
        msg = MIMEText(body)
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECIPIENT_EMAIL
        msg["Subject"] = ""

        # Connect to the SMTP server and send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        server.quit()

        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Example usage
if __name__ == "__main__":
    send_email("", "Ring ring banana phone")
    
    