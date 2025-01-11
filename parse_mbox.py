import mailbox

def extract_emails_from_mbox(mbox_file):
    emails = []
    mbox = mailbox.mbox(mbox_file)
    for message in mbox:
        # Check if the email is multipart
        if message.is_multipart():
            for part in message.get_payload():
                if part.get_content_type() == 'text/plain':  # Extract plain-text content
                    emails.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
        else:
            # Single-part email content
            emails.append(message.get_payload(decode=True).decode('utf-8', errors='ignore'))
    return emails


mbox_file = "mail.mbox/mbox"
emails = extract_emails_from_mbox(mbox_file)
print(f"Extracted {len(emails)} emails.")
