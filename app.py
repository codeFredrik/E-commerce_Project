# Backward compatibility wrapper for Streamlit execution from root
from pathlib import Path

app_path = Path(__file__).parent / "src" / "ui" / "app.py"
exec(app_path.read_text(encoding="utf-8"))