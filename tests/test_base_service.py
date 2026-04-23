import pytest
import asyncio
from Services.base_service import BaseService
from Services.broker import MockBroker

@pytest.mark.asyncio
async def test_base_service_publish_invalid_schema():
    broker = MockBroker()
    svc = BaseService(broker)
    
    # Missing 'type' and 'topic'
    with pytest.raises(ValueError, match="Event must have a 'type'"):
        await svc.publish({"payload": {}})
        
    with pytest.raises(ValueError, match="Event must have a 'topic'"):
        await svc.publish({"type": "test", "payload": {}})

    with pytest.raises(ValueError, match="Event must have a 'payload'"):
        await svc.publish({"type": "test", "topic": "test.topic"})

@pytest.mark.asyncio
async def test_base_service_publish_valid_schema():
    broker = MockBroker()
    svc = BaseService(broker)
    
    events = []
    async def capture(m):
        events.append(m)
        
    await broker.subscribe("test.topic", capture)
    
    await svc.publish({
        "type": "test_type",
        "topic": "test.topic",
        "payload": {"data": "test"}
    })
    
    await asyncio.sleep(0.01)
    assert len(events) == 1
    assert "event_id" in events[0]
    assert "timestamp" in events[0]
