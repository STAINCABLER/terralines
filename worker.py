"""
worker.py — simple worker runner for development

Run inside a container to start an RQ worker for the 'terralines' queue.

Usage: python worker.py
"""
import os
from redis import Redis
from rq import Worker
from rq.connections import Connection

redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
conn = Redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        qs = ['terralines']
        worker = Worker(qs)
        worker.work()
