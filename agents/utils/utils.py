import uuid
import base64

def make_toolcall_id():
    raw = uuid.uuid4().bytes  # 16-byte random UUID
    suffix = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return "tooluse_" + suffix