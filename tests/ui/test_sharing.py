import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_UI = importlib.util.find_spec('streamlit') is not None
if HAS_UI:
    from knowing_where.ui.sharing import public_url


@unittest.skipUnless(HAS_UI, 'Requires UI dependencies')
class SharingTests(unittest.TestCase):
    def test_only_current_launcher_state_produces_a_public_link(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'state.json'
            with patch.dict('os.environ', {'KNOWING_WHERE_PUBLIC_STATE': str(state)}):
                self.assertIsNone(public_url())
                state.write_text(json.dumps({'url': 'https://sample-name.trycloudflare.com'}))
                self.assertEqual(public_url(), 'https://sample-name.trycloudflare.com')
                for value in ('http://sample-name.trycloudflare.com', 'https://example.com', None):
                    state.write_text(json.dumps({'url': value}))
                    self.assertIsNone(public_url())
                state.write_text('{broken')
                self.assertIsNone(public_url())
