"""Contract, credential handling, and selected-source tests without paid requests."""

import io
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from knowing_where.providers.huggingface import (
    ABSTENTION, HuggingFaceClient, HuggingFaceError, run_live_query, source_messages,
)


class HuggingFaceTests(unittest.TestCase):
    def response(self, value):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(value).encode()
        return opener

    def test_chat_posts_documented_api_payload(self):
        opener = self.response({'choices': [{'message': {'content': 'Port Louis'}, 'finish_reason': 'stop'}],
                                'usage': {'total_tokens': 12}, 'model': 'org/model'})
        with patch('urllib.request.build_opener', return_value=opener):
            result = HuggingFaceClient('hf_test_secret').chat('org/model',
                     [{'role': 'user', 'content': 'Capital of Mauritius?'}], max_tokens=64, temperature=.3)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'https://router.huggingface.co/v1/chat/completions')
        self.assertEqual(request.get_header('Authorization'), 'Bearer hf_test_secret')
        payload = json.loads(request.data)
        self.assertEqual(payload['max_tokens'], 64)
        self.assertFalse(payload['stream'])
        self.assertNotIn('hf_test_secret', request.data.decode())
        self.assertEqual(result['text'], 'Port Louis')
        self.assertEqual(result['usage'], {'total_tokens': 12})

    def test_model_catalogue_is_public_and_deduplicated(self):
        opener = self.response({'data': [{'id': 'z/model'}, {'id': 'a/model'}, {'id': 'a/model'}, {}]})
        with patch('urllib.request.build_opener', return_value=opener):
            models = HuggingFaceClient('hf_secret').list_models()
        self.assertEqual(models, ['a/model', 'z/model'])
        self.assertIsNone(opener.open.call_args.args[0].get_header('Authorization'))

    def test_missing_token_makes_no_network_request(self):
        with patch('urllib.request.build_opener') as opener:
            with self.assertRaisesRegex(HuggingFaceError, 'token'):
                HuggingFaceClient().chat('org/model', [{'role': 'user', 'content': 'hi'}])
            opener.assert_not_called()

    def test_remote_error_body_cannot_leak_credentials_or_prompt(self):
        for code in (400, 401, 402, 403, 404, 422, 429, 500, 503, 302):
            opener = MagicMock()
            opener.open.side_effect = urllib.error.HTTPError('https://router.huggingface.co', code,
                'hf_test_secret private prompt', {}, io.BytesIO(b'hf_test_secret private prompt'))
            with patch('urllib.request.build_opener', return_value=opener):
                with self.assertRaises(HuggingFaceError) as caught:
                    HuggingFaceClient('hf_test_secret').chat('org/model', [{'role': 'user', 'content': 'hi'}])
            self.assertIn(f'HTTP {code}', str(caught.exception))
            self.assertNotIn('hf_test_secret', str(caught.exception))
            self.assertNotIn('private prompt', str(caught.exception))

    def test_timeouts_are_not_retried(self):
        opener = MagicMock()
        opener.open.side_effect = TimeoutError('internal detail')
        with patch('urllib.request.build_opener', return_value=opener):
            with self.assertRaisesRegex(HuggingFaceError, 'timed out'):
                HuggingFaceClient('hf_secret').chat('org/model', [{'role': 'user', 'content': 'hi'}])
        self.assertEqual(opener.open.call_count, 1)

    def test_empty_reasoning_only_response_fails_cleanly(self):
        opener = self.response({'choices': [{'message': {'content': None, 'reasoning_content': 'thinking'}}]})
        with patch('urllib.request.build_opener', return_value=opener):
            with self.assertRaisesRegex(HuggingFaceError, 'no answer text'):
                HuggingFaceClient('hf_secret').chat('org/model', [{'role': 'user', 'content': 'hi'}])

    def test_malformed_response_fails_cleanly(self):
        for value in ({}, {'choices': []}, {'choices': [None]}, []):
            with patch('urllib.request.build_opener', return_value=self.response(value)):
                with self.assertRaises(HuggingFaceError):
                    HuggingFaceClient('hf_secret').chat('org/model', [{'role': 'user', 'content': 'hi'}])

    def test_invalid_request_is_rejected_before_network(self):
        client = HuggingFaceClient('hf_secret')
        with patch('urllib.request.build_opener') as opener:
            for params in ({'max_tokens': 0}, {'temperature': float('nan')}, {'max_tokens': 3000}):
                with self.assertRaises(HuggingFaceError):
                    client.chat('org/model', [{'role': 'user', 'content': 'hi'}], **params)
            opener.assert_not_called()
        with self.assertRaises(HuggingFaceError):
            HuggingFaceClient('hf_secret\ninvalid')

    def test_prompt_uses_selected_evidence_and_p_excludes_it(self):
        knowledge = source_messages('What is the limit?', 'K', 'Limit: 2,500 MUR.')
        self.assertEqual(json.loads(knowledge[1]['content'])['source'], 'K')
        self.assertIn('2,500', knowledge[1]['content'])
        parametric = source_messages('What is the limit?', 'P', 'DO NOT SEND THIS EVIDENCE')
        self.assertNotIn('DO NOT SEND', json.dumps(parametric))
        with self.assertRaisesRegex(HuggingFaceError, 'Add evidence'):
            source_messages('What is the limit?', 'W', '')

    def test_only_requested_paths_are_run_and_single_sample_has_no_agreement(self):
        client = MagicMock()
        client.chat.return_value = {'text': 'Answer', 'usage': {}, 'finish_reason': 'stop', 'model': 'org/model'}
        result = run_live_query(client, model='org/model', question='q', source='P', paths=1)
        self.assertEqual(client.chat.call_count, 1)
        self.assertIsNone(result['exact_text_agreement'])
        self.assertFalse(result['calibrated'])

    def test_multi_path_results_normalize_only_case_and_whitespace(self):
        client = MagicMock()
        client.chat.side_effect = [{'text': text, 'usage': {}} for text in ('Port Louis', 'port  louis', 'Other')]
        result = run_live_query(client, model='org/model', question='q', source='K', evidence='e', paths=3)
        self.assertEqual(client.chat.call_count, 3)
        self.assertAlmostEqual(result['exact_text_agreement'], 2 / 3)
        self.assertEqual(result['answer'], 'Port Louis')

    def test_model_abstention_is_not_a_confident_answer(self):
        client = MagicMock()
        client.chat.return_value = {'text': ABSTENTION, 'usage': {}}
        result = run_live_query(client, model='org/model', question='q', source='P')
        self.assertTrue(result['model_abstained'])

    def test_partial_failure_stops_without_making_remaining_calls(self):
        client = MagicMock()
        client.chat.side_effect = [{'text': 'a', 'usage': {}}, HuggingFaceError('rate limited')]
        with self.assertRaisesRegex(HuggingFaceError, '1 of 3 samples completed'):
            run_live_query(client, model='org/model', question='q', source='P', paths=3)
        self.assertEqual(client.chat.call_count, 2)
