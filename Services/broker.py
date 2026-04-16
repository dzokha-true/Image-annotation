import asyncio
import json
import logging

logger = logging.getLogger(__name__)

class BaseBroker:
    async def publish(self, topic: str, message: dict):
        raise NotImplementedError("Subclasses must implement publish()")
        
    async def subscribe(self, topic: str, handler):
        raise NotImplementedError("Subclasses must implement subscribe()")

class RedisBroker(BaseBroker):
    def __init__(self, redis_url="redis://localhost"):
        try:
            import redis.asyncio as redis
            self.redis = redis.from_url(redis_url)
        except ImportError:
            logger.warning("redis package not installed, RedisBroker will fail.")

        self.pubsub = self.redis.pubsub()
        self.handlers = {}

    async def publish(self, topic: str, message: dict):
        await self.redis.publish(topic, json.dumps(message))
        
    async def subscribe(self, topic: str, handler):
        if topic not in self.handlers:
            self.handlers[topic] = []
        self.handlers[topic].append(handler)

        await self.pubsub.subscribe(**{topic: self._message_handler})
        
    async def _message_handler(self, message):
        if message and message['type'] == 'message':
            # Get topic and data from message
            topic = message['channel'].decode('utf-8')

            if topic in self.handlers:
                data = json.loads(message['data'].decode('utf-8'))
                for handler in self.handlers[topic]:
                    # create a task for each handler with the data that has arrived
                    asyncio.create_task(handler(data))

    async def start_listening(self):
        if self.pubsub is None:
            logger.warning("Pubsub not initialized. Call subscribe first.")
            return

        # Listen for messages    
        async for message in self.pubsub.listen():
            await self._message_handler(message)

class MockBroker(BaseBroker):
    def __init__(self):
        self.topics = {}
        
    async def publish(self, topic: str, message: dict):
        if topic in self.topics:
            for handler in self.topics[topic]:
                asyncio.create_task(handler(message))
                
    async def subscribe(self, topic: str, handler):
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(handler)
