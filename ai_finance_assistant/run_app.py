import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.web_app.app import FinancialAgentsApp

if __name__ == "__main__":
    app = FinancialAgentsApp()
    app.run()
