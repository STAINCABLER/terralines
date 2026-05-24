from __future__ import annotations

import copy
import unittest

from app import app
from generator import DEFAULT_PARAMS


class TerralinesAppTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def _preview_params(self) -> dict:
        params = copy.deepcopy(DEFAULT_PARAMS)
        params.update(
            {
                'width': 320,
                'height': 180,
                'preview_scale': 0.1,
                'octaves': 1,
                'levels': 2,
                'smoothing': 0.0,
                'seed': 42,
            }
        )
        return params

    def test_index_page_renders_and_sets_security_headers(self):
        response = self.client.get('/')

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn('Topografischer Mustergenerator', body)
        self.assertIn('Made by Tobias Maimone of LTM-Labs', body)
        self.assertEqual(response.headers['X-Frame-Options'], 'DENY')
        self.assertEqual(response.headers['Referrer-Policy'], 'no-referrer')
        self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy'])

    def test_defaults_and_presets_endpoints_return_json(self):
        defaults_response = self.client.get('/api/defaults')
        presets_response = self.client.get('/api/presets')

        self.assertEqual(defaults_response.status_code, 200)
        self.assertEqual(presets_response.status_code, 200)

        defaults = defaults_response.get_json()
        presets = presets_response.get_json()

        self.assertIsInstance(defaults, dict)
        self.assertIn('width', defaults)
        self.assertIn('height', defaults)
        self.assertIn('seed', defaults)
        self.assertIsInstance(presets, dict)
        self.assertGreater(len(presets), 0)

    def test_generate_preview_returns_image_metadata(self):
        response = self.client.post('/api/generate', json=self._preview_params())

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()

        self.assertIsInstance(payload, dict)
        self.assertIn('image', payload)
        self.assertIn('width', payload)
        self.assertIn('height', payload)
        self.assertIn('time_ms', payload)
        self.assertEqual(payload['width'], 320)
        self.assertEqual(payload['height'], 180)

    def test_generate_rejects_foreign_origin(self):
        response = self.client.post(
            '/api/generate',
            json=self._preview_params(),
            headers={'Origin': 'https://example.com'},
        )

        self.assertEqual(response.status_code, 403)

    def test_generate_accepts_forwarded_https_origin(self):
        response = self.client.post(
            '/api/generate',
            json=self._preview_params(),
            headers={
                'Origin': 'https://terralines.ptb.ltm-labs.de',
                'Host': 'terralines.ptb.ltm-labs.de',
                'X-Forwarded-Proto': 'https',
                'X-Forwarded-Host': 'terralines.ptb.ltm-labs.de',
                'X-Forwarded-For': '203.0.113.10',
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIn('image', payload)


if __name__ == '__main__':
    unittest.main()