"""
Database Module with OpenAI Function Calling Support
Manages SQLite database for interview scheduling.
Provides tools for LangChain agents.
"""

import sqlite3
import json
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

DB_PATH = "tech.db"


class DatabaseManager:
    """
    Manages the recruitment database and provides function calling tools.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_available_slots(
        self,
        date: Optional[str] = None,
        position: str = "Python Dev",
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves available interview slots.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')

        if date:
            query = """
                SELECT ScheduleID as id, date, time, position FROM Schedule
                WHERE available = 1 AND position = ? AND date = ?
                ORDER BY time LIMIT ?
            """
            cursor.execute(query, (position, date, limit))
        else:
            query = """
                SELECT ScheduleID as id, date, time, position FROM Schedule
                WHERE available = 1 AND position = ? AND date >= ?
                ORDER BY date, time LIMIT ?
            """
            cursor.execute(query, (position, today, limit))

        slots = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return slots

    def check_slot_availability(
        self, date: str, time: str, position: str = "Python Dev"
    ) -> bool:
        """Checks if a specific slot is available."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT available FROM Schedule
            WHERE date = ? AND time = ? AND position = ?
        """, (date, time, position))
        result = cursor.fetchone()
        conn.close()
        return result is not None and result["available"] == 1

    def book_slot(
        self, date: str, time: str, position: str = "Python Dev"
    ) -> Dict[str, Any]:
        """Books an interview slot."""
        if not self.check_slot_availability(date, time, position):
            return {"success": False, "message": f"Slot {date} at {time} is not available"}

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE Schedule SET available = 0
            WHERE date = ? AND time = ? AND position = ?
        """, (date, time, position))
        conn.commit()
        conn.close()

        return {
            "success": True,
            "message": f"Successfully booked interview for {date} at {time}",
            "date": date, "time": time, "position": position
        }

    def get_slots_near_date(
        self, target_date: str, position: str = "Python Dev", limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Finds available slots near a target date."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ScheduleID as id, date, time, position,
                   ABS(julianday(date) - julianday(?)) as date_diff
            FROM Schedule
            WHERE available = 1 AND position = ?
            ORDER BY date_diff, time LIMIT ?
        """, (target_date, position, limit))
        slots = [{"id": r["id"], "date": r["date"], "time": r["time"], "position": r["position"]}
                 for r in cursor.fetchall()]
        conn.close()
        return slots


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
    db = DatabaseManager()

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


def init_database(db_path: str = DB_PATH):
    """Initializes the recruitment database with sample data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS Schedule')
    cursor.execute('''
        CREATE TABLE Schedule (
            ScheduleID INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            position TEXT NOT NULL,
            available INTEGER NOT NULL
        )
    ''')

    positions = ['Python Dev', 'Sql Dev', 'Analyst', 'ML']
    start_date = datetime.now()
    rows = 0

    for i in range(30):
        current_date = start_date + timedelta(days=i)
        if current_date.weekday() in [0, 5]:  # Skip Saturday & Monday
            continue
        date_str = current_date.strftime('%Y-%m-%d')
        for hour in range(9, 17):
            time_str = f"{hour:02d}:00"
            for pos in positions:
                is_available = 1 if random.random() > 0.5 else 0
                cursor.execute('''
                    INSERT INTO Schedule (date, time, position, available)
                    VALUES (?, ?, ?, ?)
                ''', (date_str, time_str, pos, is_available))
                rows += 1

    conn.commit()
    conn.close()
    print(f"Database initialized with {rows} slots.")


if __name__ == "__main__":
    init_database()
