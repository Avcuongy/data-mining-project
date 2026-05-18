import os
from dotenv import load_dotenv

load_dotenv()

import warnings

warnings.filterwarnings("ignore")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
