"""
MongoDB Database Manager for Interview Scheduling
Connects to MongoDB Atlas to manage interview availability slots.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Try to import pymongo, fall back to mock if not available
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("Warning: pymongo not installed. Using mock data.")

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = "Project"
COLLECTION_NAME = "Availability"


class MongoDBManager:
    """
    Manages the recruitment database using MongoDB.
    Provides function calling tools for the Scheduling Advisor.
    """

    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._mock_data = self._load_mock_data()
        
        if MONGO_AVAILABLE and MONGO_URI:
            try:
                self.client = MongoClient(MONGO_URI)
                self.db = self.client[DB_NAME]
                self.collection = self.db[COLLECTION_NAME]
                # Test connection
                self.client.admin.command('ping')
                print("MongoDB connection successful!")
            except Exception as e:
                print(f"MongoDB connection failed: {e}. Using mock data.")
                self.client = None

    def _load_mock_data(self) -> List[Dict]:
        """Load mock data from availability.json for fallback."""
        try:
            with open("availability.json", "r") as f:
                return json.load(f)
        except:
            # Generate mock slots if file doesn't exist
            slots = []
            today = datetime.now()
            for i in range(1, 8):
                date = today + timedelta(days=i)
                if date.weekday() < 5:  # Weekdays only
                    for hour in [9, 10, 11, 14, 15, 16]:
                        slots.append({
                            "date": date.strftime("%Y-%m-%d"),
                            "time": f"{hour:02d}:00",
                            "position": "Python Dev",
                            "available": True
                        })
            return slots

    def get_available_slots(
        self,
        date: Optional[str] = None,
        position: str = "Python Dev",
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves available interview slots from MongoDB.
        """
        if self.collection is not None:
            try:
                query = {"available": True, "position": position}
                if date:
                    query["date"] = date
                else:
                    query["date"] = {"$gte": datetime.now().strftime("%Y-%m-%d")}
                
                cursor = self.collection.find(query).sort("date", 1).limit(limit)
                return [{"date": doc["date"], "time": doc["time"], "position": doc["position"]} 
                        for doc in cursor]
            except Exception as e:
                print(f"MongoDB query error: {e}")
        
        # Fallback to mock data
        today = datetime.now().strftime("%Y-%m-%d")
        slots = [s for s in self._mock_data 
                 if s["available"] and s["position"] == position 
                 and s["date"] >= today]
        if date:
            slots = [s for s in slots if s["date"] == date]
        return slots[:limit]

    def check_slot_availability(
        self, date: str, time: str, position: str = "Python Dev"
    ) -> bool:
        """Checks if a specific slot is available."""
        if self.collection is not None:
            try:
                doc = self.collection.find_one({
                    "date": date, "time": time, "position": position
                })
                return doc is not None and doc.get("available", False)
            except Exception as e:
                print(f"MongoDB query error: {e}")
        
        # Fallback
        for slot in self._mock_data:
            if slot["date"] == date and slot["time"] == time and slot["position"] == position:
                return slot["available"]
        return False

    def book_slot(
        self, date: str, time: str, position: str = "Python Dev"
    ) -> Dict[str, Any]:
        """Books an interview slot."""
        if not self.check_slot_availability(date, time, position):
            return {"success": False, "message": f"Slot {date} at {time} is not available"}

        if self.collection is not None:
            try:
                self.collection.update_one(
                    {"date": date, "time": time, "position": position},
                    {"$set": {"available": False}}
                )
                return {
                    "success": True,
                    "message": f"Successfully booked interview for {date} at {time}",
                    "date": date, "time": time, "position": position
                }
            except Exception as e:
                print(f"MongoDB update error: {e}")

        # Fallback - update mock data
        for slot in self._mock_data:
            if slot["date"] == date and slot["time"] == time and slot["position"] == position:
                slot["available"] = False
                return {
                    "success": True,
                    "message": f"Successfully booked interview for {date} at {time}",
                    "date": date, "time": time, "position": position
                }
        
        return {"success": False, "message": "Slot not found"}

    def get_slots_near_date(
        self, target_date: str, position: str = "Python Dev", limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Finds available slots near a target date."""
        if self.collection is not None:
            try:
                # Get slots around the target date
                cursor = self.collection.find({
                    "available": True,
                    "position": position
                }).sort("date", 1).limit(limit * 3)
                
                slots = list(cursor)
                # Sort by proximity to target date
                slots.sort(key=lambda x: abs(
                    (datetime.strptime(x["date"], "%Y-%m-%d") - 
                     datetime.strptime(target_date, "%Y-%m-%d")).days
                ))
                return [{"date": s["date"], "time": s["time"], "position": s["position"]} 
                        for s in slots[:limit]]
            except Exception as e:
                print(f"MongoDB query error: {e}")
        
        # Fallback
        slots = [s for s in self._mock_data if s["available"] and s["position"] == position]
        slots.sort(key=lambda x: abs(
            (datetime.strptime(x["date"], "%Y-%m-%d") - 
             datetime.strptime(target_date, "%Y-%m-%d")).days
        ))
        return slots[:limit]


# OpenAI Function Calling Tool Definitions
SCHEDULING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": "Get available interview time slots. Call when candidate wants to schedule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format (optional)"},
                    "position": {"type": "string", "enum": ["Python Dev", "Sql Dev", "Analyst", "ML"]},
                    "limit": {"type": "integer", "description": "Max slots to return (default: 3)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_slot_availability",
            "description": "Check if a specific time slot is available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Time in HH:MM format"},
                    "position": {"type": "string", "enum": ["Python Dev", "Sql Dev", "Analyst", "ML"]}
                },
                "required": ["date", "time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_slot",
            "description": "Book an interview slot after candidate confirms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Time in HH:MM format"},
                    "position": {"type": "string", "enum": ["Python Dev", "Sql Dev", "Analyst", "ML"]}
                },
                "required": ["date", "time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_slots_near_date",
            "description": "Find available slots near a date. Use for 'next Friday', 'tomorrow', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_date": {"type": "string", "description": "Target date in YYYY-MM-DD format"},
                    "position": {"type": "string", "enum": ["Python Dev", "Sql Dev", "Analyst", "ML"]},
                    "limit": {"type": "integer"}
                },
                "required": ["target_date"]
            }
        }
    }
]


def execute_function_call(function_name: str, arguments: Dict[str, Any]) -> str:
    """Executes a function call and returns JSON result."""
    db = MongoDBManager()

    if function_name == "get_available_slots":
        result = db.get_available_slots(
            arguments.get("date"), arguments.get("position", "Python Dev"), arguments.get("limit", 3))
    elif function_name == "check_slot_availability":
        result = db.check_slot_availability(
            arguments["date"], arguments["time"], arguments.get("position", "Python Dev"))
    elif function_name == "book_slot":
        result = db.book_slot(
            arguments["date"], arguments["time"], arguments.get("position", "Python Dev"))
    elif function_name == "get_slots_near_date":
        result = db.get_slots_near_date(
            arguments["target_date"], arguments.get("position", "Python Dev"), arguments.get("limit", 3))
    else:
        result = {"error": f"Unknown function: {function_name}"}

    return json.dumps(result, ensure_ascii=False)


# Alias for backwards compatibility
DatabaseManager = MongoDBManager


if __name__ == "__main__":
    # Test the MongoDB connection
    db = MongoDBManager()
    slots = db.get_available_slots(limit=3)
    print("Available slots:")
    for slot in slots:
        print(f"  {slot['date']} at {slot['time']}")
