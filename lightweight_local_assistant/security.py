import os
import ctypes
import logging
import sys

logger = logging.getLogger(__name__)

# Global set to track active subprocesses for cleanup on cancellation/exit
active_subprocesses = set()
PLAN_FILE = ".gemini/local_plan.md"

def is_sandboxed() -> bool:
    """Checks if the process is running within a macOS sandbox (seatbelt)."""
    try:
        # libsandbox is available on macOS
        libsandbox = ctypes.CDLL("/usr/lib/libsandbox.1.dylib")
        # sandbox_check(pid, entity, flags) returns 1 if sandboxed
        return libsandbox.sandbox_check(os.getpid(), None, 0) == 1
    except Exception:
        return False

def cleanup_resources():
    """Cleanup active subprocesses and plan files on exit or cancellation."""
    if os.path.exists(PLAN_FILE):
        try:
            os.remove(PLAN_FILE)
            logger.info(f"Cleaned up {PLAN_FILE}")
        except Exception as e:
            logger.error(f"Error cleaning up plan file: {e}")
            
    for p in list(active_subprocesses):
        try:
            if p.returncode is None:
                p.terminate()
                logger.info(f"Terminated background process {p.pid}")
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
    active_subprocesses.clear()

def signal_handler(sig, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {sig}, cleaning up...")
    cleanup_resources()
    sys.exit(0)
