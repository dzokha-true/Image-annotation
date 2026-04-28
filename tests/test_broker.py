import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from Services.broker import RedisBroker

@pytest.mark.asyncio
async def test_redis_broker_publish():
    with patch("redis.asyncio.from_url") as mock_from_url:
        mock_redis = AsyncMock()
        mock_from_url.return_value = mock_redis
        
        broker = RedisBroker("redis://fake")
        
        message = {"hello": "world"}
        await broker.publish("test.topic", message)
        
        mock_redis.publish.assert_called_once_with("test.topic", json.dumps(message))

@pytest.mark.asyncio
async def test_redis_broker_subscribe_and_handler():
    with patch("redis.asyncio.from_url") as mock_from_url:
        mock_redis = MagicMock()
        mock_pubsub = AsyncMock()
        mock_redis.pubsub.return_value = mock_pubsub
        mock_from_url.return_value = mock_redis
        
        broker = RedisBroker("redis://fake")
        
        handled_messages = []
        async def handler(msg):
            handled_messages.append(msg)
            
        await broker.subscribe("test.topic", handler)
        mock_pubsub.subscribe.assert_called_once_with("test.topic")
        
        # Simulate message received
        mock_message = {
            'type': 'message',
            'channel': b'test.topic',
            'data': json.dumps({"payload": "data"}).encode('utf-8')
        }
        
        await broker._message_handler(mock_message)
        await asyncio.sleep(0.01) # let the handler task run
        
        assert len(handled_messages) == 1
        assert handled_messages[0] == {"payload": "data"}

@pytest.mark.asyncio
async def test_redis_broker_listen():
    with patch("redis.asyncio.from_url") as mock_from_url:
        mock_redis = MagicMock()
        mock_pubsub = AsyncMock()
        mock_redis.pubsub.return_value = mock_pubsub
        mock_from_url.return_value = mock_redis
        
        # Async generator for listen()
        async def mock_listen():
            yield {
                'type': 'message',
                'channel': 'test.topic',
                'data': json.dumps({"payload": "listen_data"})
            }
            # break out to prevent infinite loop
            return
            
        mock_pubsub.listen = mock_listen
        
        broker = RedisBroker("redis://fake")
        handled_messages = []
        async def handler(msg):
            handled_messages.append(msg)
            
        await broker.subscribe("test.topic", handler)
        await broker.start_listening()
        await asyncio.sleep(0.01)
        
        assert len(handled_messages) == 1
        assert handled_messages[0] == {"payload": "listen_data"}
