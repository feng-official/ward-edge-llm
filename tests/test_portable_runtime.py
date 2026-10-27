"""Test the portable request builder and transport with a simulated HTTP response."""
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ward_cli_test', ROOT / 'scripts/ward_cli.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class PortableRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.event = json.loads((ROOT / 'examples/event.json').read_text(encoding='utf-8'))

    def test_invalid_input_is_rejected(self):
        for confidence in [True, -0.1, 1.1, float('nan'), '93%']:
            with self.subTest(confidence=confidence):
                self.event['context']['confidence'] = confidence
                with self.assertRaises(ValueError):
                    cli.validate_event(self.event)

    def test_offline_task_preserves_network_semantics(self):
        self.event['context']['network_state'] = 'connected'
        with self.assertRaises(ValueError):
            cli.messages_for('offline_decision', json.dumps(self.event))

    def test_original_prompt_shape(self):
        messages = cli.messages_for('cloud_judgment', json.dumps(self.event))
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])
        self.assertIn('优先级: P1', messages[1]['content'])
        self.assertIn('床位: B07', messages[1]['content'])
        self.assertIn('judgment|confidence|advice', messages[1]['content'])

    def test_transport_builds_expected_request_without_credentials(self):
        received = []
        response = json.dumps({'choices': [{'message': {'content': 'escalate|0.93|人工复核'}}]}, ensure_ascii=False).encode('utf-8')

        def fake_urlopen(request, timeout):
            received.append((request.full_url, json.loads(request.data), request.get_header('Authorization')))
            return BytesIO(response)

        argv = ['ward_cli.py', '--task', 'cloud_judgment', '--input', str(ROOT / 'examples/event.json'),
                '--base-url', 'http://127.0.0.1:8000/v1']
        output = StringIO()
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'QWEN_API_KEY': ''}), \
             patch.object(cli.urllib.request, 'urlopen', fake_urlopen), redirect_stdout(output):
            result = cli.main()
        self.assertEqual(result, 0)
        self.assertIn('escalate|0.93|人工复核', output.getvalue())
        self.assertEqual(received[0][0], 'http://127.0.0.1:8000/v1/chat/completions')
        self.assertEqual(received[0][1]['model'], 'qwen2.5-1.5b-ward-v4-q6')
        self.assertIsNone(received[0][2])


if __name__ == '__main__':
    unittest.main()
