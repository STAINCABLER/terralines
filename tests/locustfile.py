from locust import HttpUser, task, between
import random
from generator import DEFAULT_PARAMS


class GenerateUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(5)
    def generate_preview(self):
        params = DEFAULT_PARAMS.copy()
        params['preview_scale'] = 0.7
        params['seed'] = random.randint(1, 2**31 - 1)
        # Request preview (synchronous fast path)
        with self.client.post('/api/generate', json=params, timeout=120, catch_response=True) as resp:
            if resp.status_code == 429:
                resp.failure('Server reported busy (429)')
            elif resp.status_code >= 500:
                resp.failure(f'Server error {resp.status_code}')
