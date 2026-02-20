# ============================================================================
# IMPORTANT: Inject imghdr shim BEFORE any other imports (especially Streamlit)
# This workaround is necessary for Python 3.13+ where imghdr was removed
# from the standard library. The shim must load before Streamlit's CLI or
# any Streamlit module tries to import imghdr.
# ============================================================================
import sys, importlib.util, os

shim_path = os.path.join(os.path.dirname(__file__), "imghdr.py")
if "imghdr" not in sys.modules and os.path.exists(shim_path):
    spec = importlib.util.spec_from_file_location("imghdr", shim_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["imghdr"] = module

# NOW we can safely import the actual dashboard code
# (which itself imports streamlit and all its dependencies)
from dashboard import *  # noqa: F401, F403
