from .events import BaseEvent
from .broker import BaseBroker
import logging

logger = logging.getLogger(__name__)

class BaseService:
    def __init__(self, broker: BaseBroker):
        self.broker = broker
        
    async def subscribe(self, topic: str, handler):
        await self.broker.subscribe(topic, handler)
        
    async def publish(self, event_data: dict):
        try:
            event = BaseEvent(event_data) # Recommended to use Pydantic for validation and not value errors
            await self.broker.publish(event.topic, event.to_dict())
        except ValueError as e:
            logger.error(f"Event schema validation failed: {e}")
            raise
            
    async def start(self):
        pass
