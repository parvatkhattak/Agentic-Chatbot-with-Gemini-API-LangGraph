import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Annotated, Any
from pathlib import Path
import threading

import google.generativeai as genai
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)

# Data Models
class TodoItem(BaseModel):
    id: int
    task: str
    created_at: str
    completed: bool = False
    completed_at: Optional[str] = None

class ConversationMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str

class UserProfile(BaseModel):
    name: Optional[str] = None
    todos: List[TodoItem] = []
    todo_counter: int = 0
    conversation_history: List[ConversationMessage] = []

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: UserProfile
    current_task: Optional[str]

# SQLite Database Manager
class SQLiteMemoryStore:
    def __init__(self, db_path: str = "chatbot_memory.db", session_based: bool = False):
        self.db_path = db_path
        self.session_based = session_based
        self.current_user = None
        
        # Thread-local storage for database connections
        self.local = threading.local()
        
        # Initialize database
        self._init_database()
        
        # Current user profile cache
        self.current_user_profile = UserProfile()
    
    def _get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self.local, 'connection'):
            if self.session_based:
                # Use in-memory database for session-based storage
                self.local.connection = sqlite3.connect(":memory:")
            else:
                self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection.row_factory = sqlite3.Row
            
            # Initialize tables for this connection if needed
            if self.session_based:
                self._create_tables(self.local.connection)
        
        return self.local.connection
    
    def _create_tables(self, conn):
        """Create database tables"""
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                name TEXT PRIMARY KEY,
                todo_counter INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Todos table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                task TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed BOOLEAN DEFAULT FALSE,
                completed_at TEXT,
                FOREIGN KEY (user_name) REFERENCES users (name)
            )
        """)
        
        # Conversation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_name) REFERENCES users (name)
            )
        """)
        
        conn.commit()
    
    def _init_database(self):
        """Initialize the database with tables"""
        if not self.session_based:
            conn = sqlite3.connect(self.db_path)
            self._create_tables(conn)
            conn.close()
    
    def _find_existing_user(self, name: str) -> Optional[str]:
        """Find existing user by case-insensitive search"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM users WHERE LOWER(name) = LOWER(?)", (name.strip(),))
        result = cursor.fetchone()
        
        return result['name'] if result else None
    
    def switch_user(self, name: str) -> str:
        """Switch to a specific user, creating new profile if needed"""
        name = name.strip()
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if user already exists (case-insensitive)
        existing_user = self._find_existing_user(name)
        
        if existing_user:
            # Switch to existing user
            self.current_user = existing_user
            self.current_user_profile = self._load_user_profile(existing_user)
            return f"👋 Welcome back, {existing_user}! I've loaded your previous data."
        else:
            # Create new user
            cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
            conn.commit()
            
            self.current_user = name
            self.current_user_profile = UserProfile(name=name)
            return f"👋 Nice to meet you, {name}! I've created a new profile for you."
    
    def _load_user_profile(self, user_name: str) -> UserProfile:
        """Load user profile from database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get user info
        cursor.execute("SELECT * FROM users WHERE name = ?", (user_name,))
        user_row = cursor.fetchone()
        
        if not user_row:
            return UserProfile(name=user_name)
        
        # Get todos
        cursor.execute("""
            SELECT * FROM todos 
            WHERE user_name = ? 
            ORDER BY created_at DESC
        """, (user_name,))
        todo_rows = cursor.fetchall()
        
        todos = []
        for row in todo_rows:
            todos.append(TodoItem(
                id=row['id'],
                task=row['task'],
                created_at=row['created_at'],
                completed=bool(row['completed']),
                completed_at=row['completed_at']
            ))
        
        # Get conversation history (last 50 messages)
        cursor.execute("""
            SELECT * FROM conversation_history 
            WHERE user_name = ? 
            ORDER BY timestamp DESC 
            LIMIT 50
        """, (user_name,))
        conv_rows = cursor.fetchall()
        
        conversation_history = []
        for row in reversed(conv_rows):  # Reverse to get chronological order
            conversation_history.append(ConversationMessage(
                role=row['role'],
                content=row['content'],
                timestamp=row['timestamp']
            ))
        
        return UserProfile(
            name=user_name,
            todos=todos,
            todo_counter=user_row['todo_counter'],
            conversation_history=conversation_history
        )
    
    def get_user_profile(self) -> UserProfile:
        """Get current user profile"""
        if self.current_user:
            # Refresh profile from database
            self.current_user_profile = self._load_user_profile(self.current_user)
        return self.current_user_profile
    
    def add_todo(self, task: str) -> TodoItem:
        """Add a new todo item"""
        if not self.current_user:
            raise ValueError("No user selected")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Insert todo
        cursor.execute("""
            INSERT INTO todos (user_name, task, created_at) 
            VALUES (?, ?, ?)
        """, (self.current_user, task.strip(), datetime.now().isoformat()))
        
        todo_id = cursor.lastrowid
        
        # Update user's todo counter
        cursor.execute("""
            UPDATE users 
            SET todo_counter = todo_counter + 1 
            WHERE name = ?
        """, (self.current_user,))
        
        conn.commit()
        
        return TodoItem(
            id=todo_id,
            task=task.strip(),
            created_at=datetime.now().isoformat(),
            completed=False
        )
    
    def get_active_todos(self) -> List[TodoItem]:
        """Get active (incomplete) todos for current user"""
        if not self.current_user:
            return []
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM todos 
            WHERE user_name = ? AND completed = FALSE 
            ORDER BY created_at ASC
        """, (self.current_user,))
        
        todos = []
        for row in cursor.fetchall():
            todos.append(TodoItem(
                id=row['id'],
                task=row['task'],
                created_at=row['created_at'],
                completed=bool(row['completed']),
                completed_at=row['completed_at']
            ))
        
        return todos
    
    def get_completed_todos(self, limit: int = 10) -> List[TodoItem]:
        """Get completed todos for current user"""
        if not self.current_user:
            return []
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM todos 
            WHERE user_name = ? AND completed = TRUE 
            ORDER BY completed_at DESC 
            LIMIT ?
        """, (self.current_user, limit))
        
        todos = []
        for row in cursor.fetchall():
            todos.append(TodoItem(
                id=row['id'],
                task=row['task'],
                created_at=row['created_at'],
                completed=bool(row['completed']),
                completed_at=row['completed_at']
            ))
        
        return todos
    
    def complete_todo(self, task_identifier: str) -> Optional[TodoItem]:
        """Mark a todo as completed"""
        if not self.current_user:
            return None
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get active todos
        active_todos = self.get_active_todos()
        
        if not active_todos:
            return None
        
        # Clean up the task identifier
        task_identifier = task_identifier.strip().strip('"\'')
        
        # Try to find by exact match first
        for todo in active_todos:
            if task_identifier.lower() == todo.task.lower():
                cursor.execute("""
                    UPDATE todos 
                    SET completed = TRUE, completed_at = ? 
                    WHERE id = ?
                """, (datetime.now().isoformat(), todo.id))
                conn.commit()
                todo.completed = True
                todo.completed_at = datetime.now().isoformat()
                return todo
        
        # Try to find by partial match
        for todo in active_todos:
            if task_identifier.lower() in todo.task.lower():
                cursor.execute("""
                    UPDATE todos 
                    SET completed = TRUE, completed_at = ? 
                    WHERE id = ?
                """, (datetime.now().isoformat(), todo.id))
                conn.commit()
                todo.completed = True
                todo.completed_at = datetime.now().isoformat()
                return todo
        
        # Try to find by number
        try:
            task_num = int(task_identifier)
            if 1 <= task_num <= len(active_todos):
                todo = active_todos[task_num - 1]
                cursor.execute("""
                    UPDATE todos 
                    SET completed = TRUE, completed_at = ? 
                    WHERE id = ?
                """, (datetime.now().isoformat(), todo.id))
                conn.commit()
                todo.completed = True
                todo.completed_at = datetime.now().isoformat()
                return todo
        except ValueError:
            pass
        
        return None
    
    def add_conversation_message(self, role: str, content: str):
        """Add a message to conversation history"""
        if not self.current_user:
            return
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversation_history (user_name, role, content, timestamp) 
            VALUES (?, ?, ?, ?)
        """, (self.current_user, role, content, datetime.now().isoformat()))
        
        conn.commit()
        
        # Clean up old messages (keep only last 50)
        cursor.execute("""
            DELETE FROM conversation_history 
            WHERE user_name = ? AND id NOT IN (
                SELECT id FROM conversation_history 
                WHERE user_name = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            )
        """, (self.current_user, self.current_user))
        
        conn.commit()
    
    def get_all_users(self) -> List[str]:
        """Get list of all registered users"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM users ORDER BY name")
        return [row['name'] for row in cursor.fetchall()]
    
    def reset_profile(self):
        """Reset the current user profile"""
        if not self.current_user:
            return
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Delete all todos for current user
        cursor.execute("DELETE FROM todos WHERE user_name = ?", (self.current_user,))
        
        # Delete conversation history
        cursor.execute("DELETE FROM conversation_history WHERE user_name = ?", (self.current_user,))
        
        # Reset todo counter
        cursor.execute("UPDATE users SET todo_counter = 0 WHERE name = ?", (self.current_user,))
        
        conn.commit()
        
        # Reset in-memory profile
        self.current_user_profile = UserProfile(name=self.current_user)

# Tool Definitions
class AddTodoTool(BaseTool):
    name = "add_todo"
    description = "Add a new task to the user's to-do list"
    
    def _run(self, task: str) -> str:
        # Access memory store from global scope
        memory_store = globals().get('memory_store')
        if not memory_store:
            return "Error: Memory store not available."
        
        try:
            todo = memory_store.add_todo(task)
            return f"✅ Added '{task}' to your to-do list."
        except ValueError as e:
            return f"❌ Error: {str(e)}"
    
    def _arun(self, task: str) -> str:
        return self._run(task)

class ListTodosTool(BaseTool):
    name = "list_todos"
    description = "List all current to-do items"
    
    def _run(self) -> str:
        memory_store = globals().get('memory_store')
        if not memory_store:
            return "Error: Memory store not available."
        
        active_todos = memory_store.get_active_todos()
        
        if not active_todos:
            return "📝 Your to-do list is empty. Great job staying on top of things!"
        
        todo_list = "📋 Here's your current to-do list:\n"
        for i, todo in enumerate(active_todos, 1):
            created_date = datetime.fromisoformat(todo.created_at).strftime("%m/%d")
            todo_list += f"{i}. {todo.task} (added {created_date})\n"
        
        return todo_list.strip()
    
    def _arun(self) -> str:
        return self._run()

class RemoveTodoTool(BaseTool):
    name = "remove_todo"
    description = "Remove a task from the to-do list by task name or number"
    
    def _run(self, task_identifier: str) -> str:
        memory_store = globals().get('memory_store')
        if not memory_store:
            return "Error: Memory store not available."
        
        active_todos = memory_store.get_active_todos()
        
        if not active_todos:
            return "📝 Your to-do list is empty, nothing to remove."
        
        completed_todo = memory_store.complete_todo(task_identifier)
        
        if completed_todo:
            return f"✅ Completed '{completed_todo.task}' and removed it from your to-do list!"
        else:
            return f"❌ Could not find task '{task_identifier}' in your to-do list. Try listing your todos first to see what's available."
    
    def _arun(self, task_identifier: str) -> str:
        return self._run(task_identifier)

class UpdateNameTool(BaseTool):
    name = "update_name"
    description = "Update or set the user's name"
    
    def _run(self, name: str) -> str:
        memory_store = globals().get('memory_store')
        if not memory_store:
            return "Error: Memory store not available."
        
        name = name.strip()
        response = memory_store.switch_user(name)
        
        # Get active todos after switching
        active_todos = memory_store.get_active_todos()
        
        if active_todos:
            todo_list = f"\n📋 Your current to-do list:\n"
            for i, todo in enumerate(active_todos, 1):
                created_date = datetime.fromisoformat(todo.created_at).strftime("%m/%d")
                todo_list += f"{i}. {todo.task} (added {created_date})\n"
            response += todo_list.strip()
        else:
            response += "\n📝 Your to-do list is empty."
        
        return response
    
    def _arun(self, name: str) -> str:
        return self._run(name)

class GetCompletedTodosTool(BaseTool):
    name = "get_completed_todos"
    description = "Show recently completed to-do items"
    
    def _run(self) -> str:
        memory_store = globals().get('memory_store')
        if not memory_store:
            return "Error: Memory store not available."
        
        completed_todos = memory_store.get_completed_todos(10)
        
        if not completed_todos:
            return "📝 No completed tasks yet. You can do it!"
        
        todo_list = "✅ Your recently completed tasks:\n"
        for i, todo in enumerate(completed_todos, 1):
            completed_date = datetime.fromisoformat(todo.completed_at).strftime("%m/%d") if todo.completed_at else "Unknown"
            todo_list += f"{i}. {todo.task} (completed {completed_date})\n"
        
        return todo_list.strip()
    
    def _arun(self) -> str:
        return self._run()

class ListUsersTool(BaseTool):
    name = "list_users"
    description = "List all registered users"
    
    def _run(self) -> str:
        memory_store = globals().get('memory_store')
        if not memory_store:
            return "Error: Memory store not available."
        
        users = memory_store.get_all_users()
        current_user = memory_store.current_user
        
        if not users:
            return "📝 No users registered yet."
        
        user_list = "👥 Registered users:\n"
        for user in users:
            marker = " (current)" if user == current_user else ""
            user_list += f"• {user}{marker}\n"
        
        return user_list.strip()
    
    def _arun(self) -> str:
        return self._run()

# Enhanced Intent Detection using Gemini (unchanged)
class GeminiIntentDetector:
    def __init__(self):
        self.intent_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,  # Low temperature for consistent intent detection
            convert_system_message_to_human=True
        )
    
    def detect_intent(self, text: str, current_user: Optional[str] = None) -> tuple[str, str]:
        """Use Gemini to detect intent and extract relevant information"""
        
        # Create the intent detection prompt
        system_prompt = """You are an intent detection system for a personal assistant chatbot. 
        
        Analyze the user's message and determine their intent. Return ONLY a JSON response with this exact format:
        {
            "intent": "intent_name",
            "extracted_info": "relevant_information"
        }
        
        Available intents:
        - "add_todo": User wants to add a task to their to-do list
        - "list_todos": User wants to see their current to-do list
        - "remove_todo": User wants to remove/complete a task from their to-do list
        - "update_name": User wants to set/change their name or switch users
        - "get_completed_todos": User wants to see completed tasks
        - "list_users": User wants to see all registered users
        - "chat": General conversation, questions, or anything else
        
        For extracted_info:
        - add_todo: Extract the task description
        - remove_todo: Extract the task name/number/identifier
        - update_name: Extract the name
        - For other intents: Leave empty string ""
        
        Examples:
        User: "Add buy milk to my todo list" -> {"intent": "add_todo", "extracted_info": "buy milk"}
        User: "What's on my todo list?" -> {"intent": "list_todos", "extracted_info": ""}
        User: "I finished the groceries task" -> {"intent": "remove_todo", "extracted_info": "groceries"}
        User: "My name is John" -> {"intent": "update_name", "extracted_info": "John"}
        User: "How's the weather?" -> {"intent": "chat", "extracted_info": ""}
        User: "What completed tasks do I have?" -> {"intent": "get_completed_todos", "extracted_info": ""}
        User: "Show all users" -> {"intent": "list_users", "extracted_info": ""}
        
        Be flexible with natural language variations but be conservative - when in doubt, use "chat" intent."""
        
        # Add context about current user if available
        context = ""
        if current_user:
            context = f"\n\nCurrent user: {current_user}"
        
        messages = [
            SystemMessage(content=system_prompt + context),
            HumanMessage(content=f"User message: {text}")
        ]
        
        try:
            response = self.intent_llm.invoke(messages)
            response_text = response.content.strip()
            
            # Clean up the response to extract JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
            
            # Parse the JSON response
            import json
            intent_data = json.loads(response_text)
            intent = intent_data.get("intent", "chat")
            extracted_info = intent_data.get("extracted_info", "")
            
            return intent, extracted_info
            
        except Exception as e:
            print(f"Intent detection error: {e}")
            # Fallback to chat intent if there's any error
            return "chat", ""

# Enhanced Agent Architecture
class AgenticChatbot:
    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.intent_detector = GeminiIntentDetector()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Initialize tools
        self.tools = {
            "add_todo": AddTodoTool(),
            "list_todos": ListTodosTool(),
            "remove_todo": RemoveTodoTool(),
            "update_name": UpdateNameTool(),
            "get_completed_todos": GetCompletedTodosTool(),
            "list_users": ListUsersTool()
        }
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def chatbot_node(state: AgentState) -> AgentState:
            """Main chatbot reasoning node"""
            # Always get fresh profile from memory store
            profile = self.memory_store.get_user_profile()
            user_message = state["messages"][-1].content
            
            # Use Gemini to detect intent
            current_user = self.memory_store.current_user
            intent, extracted_info = self.intent_detector.detect_intent(user_message, current_user)
            
            # Handle different intents
            if intent == "add_todo" and extracted_info:
                result = self.tools["add_todo"]._run(extracted_info)
                # Get updated profile after tool execution
                updated_profile = self.memory_store.get_user_profile()
                return {
                    "messages": [AIMessage(content=result)],
                    "user_profile": updated_profile,
                    "current_task": state.get("current_task")
                }
            
            elif intent == "list_todos":
                result = self.tools["list_todos"]._run()
                return {
                    "messages": [AIMessage(content=result)],
                    "user_profile": profile,
                    "current_task": state.get("current_task")
                }
            
            elif intent == "remove_todo" and extracted_info:
                result = self.tools["remove_todo"]._run(extracted_info)
                # Get updated profile after tool execution
                updated_profile = self.memory_store.get_user_profile()
                return {
                    "messages": [AIMessage(content=result)],
                    "user_profile": updated_profile,
                    "current_task": state.get("current_task")
                }
            
            elif intent == "update_name" and extracted_info:
                result = self.tools["update_name"]._run(extracted_info)
                # Get updated profile after user switching
                updated_profile = self.memory_store.get_user_profile()
                return {
                    "messages": [AIMessage(content=result)],
                    "user_profile": updated_profile,
                    "current_task": state.get("current_task")
                }
            
            elif intent == "get_completed_todos":
                result = self.tools["get_completed_todos"]._run()
                return {
                    "messages": [AIMessage(content=result)],
                    "user_profile": profile,
                    "current_task": state.get("current_task")
                }
            
            elif intent == "list_users":
                result = self.tools["list_users"]._run()
                return {
                    "messages": [AIMessage(content=result)],
                    "user_profile": profile,
                    "current_task": state.get("current_task")
                }
            
            # If no specific intent detected or intent is "chat", use LLM for general chat
            system_msg = self._create_system_message(profile)
            messages = [system_msg] + state["messages"]
            
            response = self.llm.invoke(messages)
            return {
                "messages": [response],
                "user_profile": profile,
                "current_task": state.get("current_task")
            }
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("chatbot", chatbot_node)
        
        # Add edges
        workflow.set_entry_point("chatbot")
        workflow.add_edge("chatbot", END)
        
        return workflow.compile()
    
    def _create_system_message(self, profile: UserProfile) -> SystemMessage:
        """Create system message with current context"""
        name_context = f"The user's name is {profile.name}. " if profile.name else ""
        
        active_todos = self.memory_store.get_active_todos()
        todo_context = ""
        if active_todos:
            todo_context = f"Current to-do items: {', '.join([todo.task for todo in active_todos[:5]])}{'...' if len(active_todos) > 5 else ''}. "
        
        # Include recent conversation context
        recent_context = ""
        if profile.conversation_history:
            recent_msgs = profile.conversation_history[-3:]  # Last 3 messages
            recent_context = "Recent conversation context:\n"
            for msg in recent_msgs:
                recent_context += f"- {msg.role}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}\n"
        
        system_prompt = f"""You are a helpful personal assistant chatbot with multi-user support. {name_context}{todo_context}

{recent_context}

You can help users with:
- Basic conversation and questions
- Managing their personal to-do list (add, remove, list tasks)
- Remembering their name and conversation history
- Switching between different user profiles

The system automatically handles to-do management and user switching when users mention tasks or names. Be friendly, conversational, and helpful. 

If asked about your capabilities, mention that you can:
- Have natural conversations while remembering context
- Manage personal to-do lists for multiple users
- Switch between different user profiles
- Remember names and past conversations for each user
- Help with general questions and tasks

Keep responses concise and natural. Use emojis sparingly to make interactions more friendly."""
        
        return SystemMessage(content=system_prompt)
    
    def chat(self, message: str) -> str:
        """Process a user message and return response"""
        # Store user message in conversation history
        self.memory_store.add_conversation_message("user", message)
        
        # Always get fresh profile from memory store
        current_profile = self.memory_store.get_user_profile()
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_profile": current_profile,
            "current_task": None
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Get the last AI message
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            response = ai_messages[-1].content
            # Store assistant response in conversation history
            self.memory_store.add_conversation_message("assistant", response)
            return response
        
        return "I'm sorry, I didn't understand that. Can you please try again?"

# Enhanced CLI Interface
def main():
    """Main CLI interface"""
    print("🤖 Welcome to your Personal Assistant Chatbot!")
    print("I can help you with conversations and manage your to-do list.")
    print("Now with multi-user support and SQLite database persistence!")
    print("Try saying things like:")
    print("  - 'My name is John'")
    print("  - 'Add buy groceries to my todo list'")
    print("  - 'Show my todos'")
    print("  - 'Change my name to Sarah'")
    print("  - 'I finished the groceries task'")
    print("  - 'show all users'")
    print("  - Or just chat naturally!")
    print("Type 'quit' to exit\n")
    
    # Create global memory store with session_based=False for persistent storage
    global memory_store
    memory_store = SQLiteMemoryStore(db_path="chatbot_memory.db", session_based=False)
    
    # For CLI, always start fresh
    print("👋 Welcome! What's your name?")
    
    chatbot = AgenticChatbot(memory_store)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                current_user = memory_store.current_user
                farewell = f"Goodbye, {current_user}!" if current_user else "Goodbye!"
                print(f"Chatbot: {farewell} Have a great day! 👋")
                break
            
            if not user_input:
                continue
            
            response = chatbot.chat(user_input)
            print(f"Chatbot: {response}\n")
            
        except KeyboardInterrupt:
            print("\nChatbot: Goodbye! Have a great day! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()