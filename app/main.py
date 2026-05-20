import sys
import pathlib
import streamlit as st
from importlib import import_module

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PAGES = {
    "Overview": "app.pages.overview",
    "Payment Analysis": "app.pages.payment_analysis",
}


def main():
    st.set_page_config(page_title="Sales Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))

    module = import_module(PAGES[choice])
    module.page()


if __name__ == "__main__":
    main()
