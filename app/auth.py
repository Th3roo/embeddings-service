import os
from typing import List, Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

load_dotenv()

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

raw_keys = os.getenv("VALID_API_KEYS", "")
VALID_API_KEYS: List[str] = [key.strip() for key in raw_keys.replace(',', ' ').split() if key.strip()]

if not VALID_API_KEYS:
    print("WARNING: No VALID_API_KEYS found in .env. Authorization will not work.")
    VALID_API_KEYS = ["fallback_dummy_token_for_dev_only"]

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key in VALID_API_KEYS:
        return api_key
    else:
        raise HTTPException(
            status_code=403, detail="Could not validate credentials"
        )