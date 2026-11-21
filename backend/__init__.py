# Makes `backend` importable as a package for the test-suite (`import backend.main`).
# Production still runs `uvicorn main:app` from inside backend/ (main.py handles both import styles).
