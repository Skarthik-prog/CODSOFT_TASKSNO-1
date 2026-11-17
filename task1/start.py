import os, sys
from streamlit.web import cli as stcli
sys.argv = ["streamlit", "run", os.path.join(os.path.dirname(__file__), "app.py")]
stcli.main()
