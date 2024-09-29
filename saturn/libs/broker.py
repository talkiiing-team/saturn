import os
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend

dsn = os.environ.get("REDIS_DSN", "redis://localhost:6379")

broker = ListQueueBroker(url=dsn).with_result_backend(
    result_backend=RedisAsyncResultBackend(redis_url=dsn)
)
