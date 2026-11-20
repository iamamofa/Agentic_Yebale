"""Guide interactions preserve learning progress while users try real workspaces."""

import importlib.util
from pathlib import Path
import unittest

HAS_UI = importlib.util.find_spec('streamlit') is not None
if HAS_UI:
    from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(HAS_UI, 'Install UI dependencies for guide checks')
class GuideFlowTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / 'streamlit_app.py'), default_timeout=30).run()
        app.button(key='home_guides').click().run()
        self.assertFalse(app.exception)
        return app

    def test_progress_survives_workspace_navigation(self):
        app = self.app()
        app.checkbox(key='guide_done_0').check().run()
        self.assertEqual(app.progress[0].value, 20)
        app.button(key='journey_open_1').click().run()
        self.assertEqual(app.radio(key='workspace').value, 'Routing simulator')
        app.radio(key='workspace').set_value('Guides & methods').run()
        self.assertTrue(app.checkbox(key='guide_done_0').value)
        self.assertEqual(app.progress[0].value, 20)
        self.assertFalse(app.exception)

    def test_reading_lens_and_method_stages(self):
        app = self.app()
        app.segmented_control(key='guide_section').set_value('Literature review').run()
        self.assertEqual(len(app.link_button), 4)
        app.pills(key='reading_lens').set_value('Calibration').run()
        self.assertEqual(len(app.link_button), 1)
        self.assertIn('2107.07511', app.link_button[0].url)
        app.segmented_control(key='guide_section').set_value('Methodology').run()
        for stage in range(1, 7):
            app.select_slider(key='method_stage').set_value(stage).run()
            self.assertFalse(app.exception)
        app.button(key='method_open').click().run()
        self.assertEqual(app.radio(key='workspace').value, 'Uncertainty lab')
        app.radio(key='workspace').set_value('Guides & methods').run()
        self.assertEqual(app.select_slider(key='method_stage').value, 6)

    def test_plan_notes_survive_switching_guides(self):
        app = self.app()
        app.segmented_control(key='guide_section').set_value('Study planner').run()
        app.text_area(key='plan_question').set_value('Compare two retrievers on audited sources.')
        app.text_area(key='plan_notes').set_value('Group paraphrases before splitting.')
        next(b for b in app.button if b.label == 'Prepare my study brief').click().run()
        self.assertIn('Compare two retrievers on audited sources.', app.session_state['guide_prepared_brief'])
        self.assertIn('Group paraphrases before splitting.', app.session_state['guide_prepared_brief'])
        app.segmented_control(key='guide_section').set_value('First visit').run()
        app.segmented_control(key='guide_quiz').set_value('Correctness').run()
        self.assertTrue(any('shared error' in message.value for message in app.info))
        app.segmented_control(key='guide_section').set_value('Study planner').run()
        self.assertEqual(app.text_area(key='plan_notes').value, 'Group paraphrases before splitting.')
        self.assertFalse(app.exception)
