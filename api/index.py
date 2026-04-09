import sys
import os
from pathlib import Path

# Add the root directory to sys.path to resolve 'apps' and 'packages'
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

app = None

try:
    from apps.api.main import app as _app
    app = _app
except Exception as e:
    import traceback
    print("FATAL ERROR DURING IMPORT:")
    print(traceback.format_exc())
    # Re-raise so Vercel logs the failure
    raise e
