"""Show the currently running launcher's verified public link, when present."""

import json
import os
from pathlib import Path
import re

import streamlit as st


def public_url():
    state_path = os.environ.get('KNOWING_WHERE_PUBLIC_STATE')
    if not state_path:
        return None
    try:
        state = json.loads(Path(state_path).read_text())
        url = state['url']
        if not isinstance(url, str) or not re.fullmatch(r'https://[a-z0-9-]+\.trycloudflare\.com', url):
            return None
        return url
    except (OSError, ValueError, KeyError, TypeError):
        return None


def sharing_panel():
    url = public_url()
    if url:
        st.markdown('**Public app link**')
        st.link_button('Open shared app', url, use_container_width=True)
        with st.expander('Copy public link'):
            st.code(url, language=None)
        st.caption('Anyone with this link can visit while the launcher is running.')
    else:
        st.caption('To create a public link, start the app with start_app.command.')
