import uuid
from datetime import datetime, timezone

class BaseEvent:
    def __init__(self, data: dict):
        self.type = data.get("type")
        self.topic = data.get("topic")
        self.payload = data.get("payload")
        
        if self.type is None:
            raise ValueError("Event must have a 'type'")
        if self.topic is None:
            raise ValueError("Event must have a 'topic'")
        if self.payload is None:
            raise ValueError("Event must have a 'payload'")
            
        self.event_id = data.get("event_id") or str(uuid.uuid4())
        self.timestamp = data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "topic": self.topic,
            "event_id": self.event_id,
            "payload": self.payload,
            "timestamp": self.timestamp
        }
