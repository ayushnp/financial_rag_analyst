import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Supabase Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


if not GOOGLE_API_KEY:
    raise ValueError("CRITICAL: Missing GOOGLE_API_KEY in .env file")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("CRITICAL: Missing Supabase credentials in .env file")