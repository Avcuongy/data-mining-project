import os
import json
import datetime
import pandas as pd


def get_latest_file_in_directory(directory, extension):
    """
    Get the latest file in a directory with a specific extension.

    Args:
        directory (str): Directory to search for files.
        extension (str): File extension to look for.

    Returns:
        str or None: Path to the latest file or None if no files are found.
    """
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(extension)
    ]
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file
