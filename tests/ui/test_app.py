"""Streamlit flow tests are optional; service tests need only the core library."""

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from knowing_where.ui import services

HAS_UI = importlib.util.find_spec('streamlit') is not None
if HAS_UI:
    from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


class UIServiceTests(unittest.TestCase):
    def test_context_service_runs_existing_example(self):
        result = services.compile_context(services.sample_text('context'))
        self.assertEqual(result['selected_model_id'], 'demo-medium')
        self.assertTrue(result['candidates'][0]['context']['conflicts'])

    def test_prediction_service_returns_label_free_results(self):
        result = services.predict(services.sample_text('system'), services.sample_text('queries'), max_calls=120)
        self.assertEqual(len(result['predictions']), 3)
        self.assertNotIn('gold', json.dumps(result))

    def test_upload_validation_and_line_errors(self):
        with self.assertRaisesRegex(ValueError, 'Line 2'):
            services.jsonl_input(services.sample_text('queries').splitlines()[0] + '\n{broken}', labeled=False)
        with self.assertRaisesRegex(ValueError, 'UTF-8'):
            services.decode(None)
        self.assertEqual(services.decode(b'\xef\xbb\xbf{}'), '{}')
        with self.assertRaisesRegex(ValueError, 'JSON object'):
            services.predict('[]', services.sample_text('queries'), max_calls=120)
        with self.assertRaisesRegex(ValueError, 'unique IDs'):
            line = services.sample_text('queries').splitlines()[0]
            services.jsonl_input(line + '\n' + line, labeled=False)

    def test_oversized_upload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '10 MB'):
            services.decode(b'x' * (services.MAX_BYTES + 1))


@unittest.skipUnless(HAS_UI, 'Install the optional ui dependencies to run Streamlit tests')
class StreamlitFlowTests(unittest.TestCase):
    def app(self, page=None):
        app = AppTest.from_file(str(ROOT / 'streamlit_app.py'), default_timeout=30).run()
        if page:
            app.radio(key='workspace').set_value(page).run()
        self.assertFalse(app.exception)
        return app

    def button(self, app, label):
        return next(button for button in app.button if button.label == label)

    def test_overview_renders_and_all_workspaces_open(self):
        app = self.app('Overview')
        self.assertIn('evidence trail', app.title[0].value)
        self.assertEqual(len(app.metric), 4)
        for page in ('Start here', 'Live models', 'Routing simulator', 'Algorithm explorer', 'Research notebook',
                     'Context studio', 'Experiments', 'Uncertainty lab', 'Predict'):
            app.radio(key='workspace').set_value(page).run()
            self.assertFalse(app.exception)

    def test_live_generation_requires_token_without_network(self):
        with patch('knowing_where.ui.live.configured_token', return_value=''):
            app = self.app('Live models')
            self.button(app, 'Generate live answer').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('token' in message.value for message in app.error))
        self.assertNotIn('hf_live_result', app.session_state)

    def test_live_model_response_stale_inputs_and_forgetting(self):
        response = {'text': '2,500 MUR per approved trip.', 'model': 'org/model',
                    'usage': {'total_tokens': 30}, 'finish_reason': 'stop'}
        with patch('knowing_where.providers.huggingface.HuggingFaceClient.chat', return_value=response) as chat:
            app = self.app('Live models')
            app.text_input(key='hf_token_input').set_value('hf_test_secret').run()
            self.button(app, 'Generate live answer').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(chat.call_count, 1)
            result = app.session_state['hf_live_result']
            self.assertEqual(result['source'], 'K')
            self.assertNotIn('hf_test_secret', json.dumps(result))
            self.assertEqual(result['answer'], response['text'])
            self.assertIsNone(result['exact_text_agreement'])
            app.text_area(key='hf_question').set_value('A different question?').run()
            self.assertEqual(chat.call_count, 1)
            self.assertTrue(any('Inputs changed' in info.value for info in app.info))
            self.button(app, 'Forget session token and live results').click().run()
            self.assertFalse(app.exception)
            self.assertNotIn('hf_live_result', app.session_state)
            self.assertEqual(app.text_input(key='hf_token_input').value, '')

    def test_live_model_catalogue_and_connection_test(self):
        with patch('knowing_where.providers.huggingface.HuggingFaceClient.list_models',
                   return_value=['test/model-a', 'test/model-b']):
            app = self.app('Live models')
            self.button(app, 'Load available models').click().run()
        self.assertFalse(app.exception)
        app.selectbox(key='hf_model_choice').set_value('test/model-b').run()
        app.text_input(key='hf_token_input').set_value('hf_test_secret').run()
        with patch('knowing_where.providers.huggingface.HuggingFaceClient.chat',
                   return_value={'text': 'connected', 'usage': {}, 'finish_reason': 'stop'}) as chat:
            self.button(app, 'Test model connection').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(chat.call_args.args[0], 'test/model-b')
        self.assertTrue(any('Live response received' in item.value for item in app.success))

    def test_failed_live_run_clears_previous_result(self):
        from knowing_where.providers.huggingface import HuggingFaceError
        with patch('knowing_where.providers.huggingface.HuggingFaceClient.chat',
                   return_value={'text': 'answer', 'model': 'test/model', 'finish_reason': 'stop', 'usage': {}}):
            app = self.app('Live models')
            app.text_input(key='hf_token_input').set_value('hf_test_secret').run()
            self.button(app, 'Generate live answer').click().run()
        with patch('knowing_where.providers.huggingface.HuggingFaceClient.chat',
                   side_effect=HuggingFaceError('rate limited')):
            self.button(app, 'Generate live answer').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertNotIn('hf_live_result', app.session_state)

    def test_beginner_tour_navigates_to_source_first_simulation(self):
        app = self.app()
        self.assertIn('knowing where to look', app.title[0].value)
        self.button(app, 'Open Routing simulator').click().run()
        self.assertEqual(app.radio(key='workspace').value, 'Routing simulator')
        self.assertFalse(app.exception)
        app.slider(key='route_stage').set_value(5).run()
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label == 'Teaching evidence score' for m in app.metric))
        app.slider(key='route_threshold').set_value(1.0).run()
        self.assertTrue(any('abstain' in message.value for message in app.warning))
        app.selectbox(key='route_scenario').set_value('A recent public event').run()
        self.assertFalse(app.exception)
        selected = next(m for m in app.metric if m.label == 'Selected source')
        self.assertEqual(selected.value, 'W')

    def test_interactive_lesson_separates_coverage_from_skill(self):
        app = self.app()
        app.segmented_control(key='home_section').set_value('Try it yourself').run()
        app.slider(key='lesson_absence').set_value(40).run()
        app.slider(key='lesson_skill').set_value(100).run()
        values = {m.label: m.value for m in app.metric}
        self.assertEqual(values['Found answers'], '60 / 100')
        self.assertEqual(values['Absent from source'], '40 / 100')
        app.slider(key='lesson_skill').set_value(0).run()
        values = {m.label: m.value for m in app.metric}
        self.assertEqual(values['Present but missed'], '60 / 100')
        self.assertEqual(values['Absent from source'], '40 / 100')
        self.assertFalse(app.exception)
        app.segmented_control(key='home_section').set_value('How to use the app').run()
        self.button(app, 'Open Context studio').click().run()
        self.assertEqual(app.radio(key='workspace').value, 'Context studio')
        self.assertFalse(app.exception)

    def test_router_skips_all_sources_when_cost_exceeds_value(self):
        app = self.app('Routing simulator')
        app.selectbox(key='route_scenario').set_value('Missing everywhere').run()
        app.slider(key='route_penalty').set_value(1.0).run()
        app.slider(key='route_stage').set_value(5).run()
        self.assertFalse(app.exception)
        selected = next(m for m in app.metric if m.label == 'Selected source')
        self.assertEqual(selected.value, 'Abstain')
        self.assertFalse(any(m.label == 'Teaching evidence score' for m in app.metric))

    def test_research_snapshots_stay_separate(self):
        app = self.app('Research notebook')
        app.radio(key='research_snapshot').set_value('Supplied PDF · progress report').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('1.5B' in element.value for element in app.markdown))

    def test_context_compile_shows_conflicts_and_detects_input_changes(self):
        app = self.app('Context studio')
        self.button(app, 'Compile context').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['context_result']['selected_model_id'], 'demo-medium')
        self.assertTrue(app.session_state['context_result']['candidates'][0]['context']['conflicts'])
        app.radio(key='context_input_mode').set_value('Paste').run()
        request = json.loads(services.sample_text('context'))
        request['question'] = 'A revised question?'
        app.text_area(key='context_input_text').set_value(json.dumps(request)).run()
        self.assertTrue(any('Inputs changed' in info.value for info in app.info))

    def test_invalid_pasted_context_displays_error_without_crashing(self):
        app = self.app('Context studio')
        app.radio(key='context_input_mode').set_value('Paste').run()
        app.text_area(key='context_input_text').set_value('{broken}').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        app.text_area(key='context_input_text').set_value('{"question":"q", "policy":[]}').run()
        self.assertFalse(app.exception)
        self.assertIn('Policy must be', app.error[0].value)

    def test_experiment_requires_explicit_independence_choice(self):
        app = self.app('Experiments')
        self.button(app, 'Run experiment').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('independence' in item.value for item in app.warning))
        self.assertNotIn('experiment_report', app.session_state)

    def test_run_experiment_export_then_predict_with_latest_system(self):
        app = self.app('Experiments')
        app.checkbox[0].check()
        self.button(app, 'Run experiment').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.session_state['experiment_report']['generation']['attempted_calls'], 4000)
        self.assertIn('fitted_system', app.session_state)
        app.radio(key='workspace').set_value('Predict').run()
        self.assertEqual(app.selectbox[0].value, 'Latest experiment')
        self.button(app, 'Run predictions').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.session_state['prediction_result']['predictions']), 3)
        self.button(app, 'Clear session results').click().run()
        self.assertNotIn('prediction_result', app.session_state)
        self.assertNotIn('fitted_system', app.session_state)

    def test_prediction_call_budget_failure_is_actionable(self):
        app = self.app('Predict')
        app.number_input[0].set_value(40).run()
        self.button(app, 'Run predictions').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('planned calls' in error.value for error in app.error))
        self.assertNotIn('prediction_result', app.session_state)

    def test_experiment_budget_failure_does_not_keep_a_fitted_system(self):
        app = self.app('Experiments')
        app.checkbox[0].check()
        next(widget for widget in app.number_input if widget.label == 'Total call budget').set_value(100)
        self.button(app, 'Run experiment').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('planned calls' in error.value for error in app.error))
        self.assertNotIn('fitted_system', app.session_state)

    def test_uncertainty_lab_separates_ambiguity_and_disagreement(self):
        app = self.app('Uncertainty lab')
        self.assertEqual(app.title[0].value, 'Know what you don’t know.')
        self.assertTrue(any(m.label == 'Leading score' for m in app.metric))
        app.radio(key='uncertainty_mode').set_value('Controlled examples').run()
        app.selectbox(key='uncertainty_preset').set_value('Each source is split').run()
        self.assertFalse(app.exception)
        disagreement = next(m for m in app.metric if m.label == 'Between-source disagreement')
        self.assertEqual(disagreement.value, '0.0%')
        app.selectbox(key='uncertainty_preset').set_value('Sources disagree').run()
        self.assertFalse(app.exception)
        disagreement = next(m for m in app.metric if m.label == 'Between-source disagreement')
        self.assertEqual(disagreement.value, '100.0%')

    def test_risk_review_changes_subset_without_changing_decisions(self):
        app = self.app('Uncertainty lab')
        before = next(m.value for m in app.metric if m.label == 'Questions in review subset')
        app.slider(key='lab_ceiling').set_value(1.0).run()
        self.assertFalse(app.exception)
        after = next(m.value for m in app.metric if m.label == 'Questions in review subset')
        self.assertNotEqual(before, after)
