from flask import Flask, request, jsonify, session
import sqlite3
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TypedDict
import google.generativeai as genai
from pathlib import Path
import json
import threading

# LangChain imports
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.pydantic_v1 import Field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Todo:
    id: int
    task: str
    completed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

@dataclass
class UserProfile:
    name: Optional[str] = None
    todos: List[Todo] = field(default_factory=list)
    conversation_history: List[Message] = field(default_factory=list)
    _todo_counter: int = 0


class SQLiteMemoryStore:
    def __init__(self, db_path="webapp_memory.db"):
        self.db_path = db_path
        self.current_user_profile = UserProfile()
        self.current_user_name = None
        self._lock = threading.Lock()
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    name TEXT PRIMARY KEY,
                    todo_counter INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create todos table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS todos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    user_todo_id INTEGER NOT NULL,
                    task TEXT NOT NULL,
                    completed BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    FOREIGN KEY (user_name) REFERENCES users (name),
                    UNIQUE(user_name, user_todo_id)
                )
            """)
            
            # Create conversation_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_name) REFERENCES users (name)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_todos_user_name ON todos(user_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_todos_completed ON todos(completed)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_user_name ON conversation_history(user_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_timestamp ON conversation_history(timestamp)")
            
            conn.commit()

    def _get_connection(self):
        """Get a database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def switch_user(self, name: str) -> str:
        """Switch to a user, creating them if they don't exist"""
        with self._lock:
            name = name.strip()
            print(f"DEBUG: switch_user called with name: {name}")
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if user exists (case-insensitive)
                cursor.execute("""
                    SELECT name FROM users WHERE LOWER(name) = LOWER(?)
                """, (name,))
                existing_user = cursor.fetchone()
                
                if existing_user:
                    existing_name = existing_user['name']
                    print(f"DEBUG: Found existing user: {existing_name}")
                    self.current_user_name = existing_name
                    self._load_user_profile(existing_name)
                    return f"Welcome back, {existing_name}!"
                else:
                    print(f"DEBUG: Creating new user: {name}")
                    # Create new user
                    cursor.execute("""
                        INSERT INTO users (name, todo_counter) VALUES (?, 0)
                    """, (name,))
                    conn.commit()
                    
                    self.current_user_name = name
                    self.current_user_profile = UserProfile(name=name)
                    return f"Hello {name}, I've created your profile."

    def _load_user_profile(self, user_name: str):
        """Load user profile from database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get user info
            cursor.execute("SELECT * FROM users WHERE name = ?", (user_name,))
            user_row = cursor.fetchone()
            
            if not user_row:
                self.current_user_profile = UserProfile()
                return
            
            # Load todos
            cursor.execute("""
                SELECT user_todo_id, task, completed, created_at, completed_at 
                FROM todos 
                WHERE user_name = ? 
                ORDER BY user_todo_id
            """, (user_name,))
            
            todos = []
            for row in cursor.fetchall():
                todo = Todo(
                    id=row['user_todo_id'],
                    task=row['task'],
                    completed=bool(row['completed']),
                    created_at=row['created_at'],
                    completed_at=row['completed_at']
                )
                todos.append(todo)
            
            # Load conversation history (last 50 messages)
            cursor.execute("""
                SELECT role, content, timestamp 
                FROM conversation_history 
                WHERE user_name = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            """, (user_name,))
            
            conversation_history = []
            for row in reversed(cursor.fetchall()):  # Reverse to get chronological order
                msg = Message(
                    role=row['role'],
                    content=row['content'],
                    timestamp=row['timestamp']
                )
                conversation_history.append(msg)
            
            self.current_user_profile = UserProfile(
                name=user_name,
                todos=todos,
                conversation_history=conversation_history,
                _todo_counter=user_row['todo_counter']
            )

    def get_user_profile(self) -> UserProfile:
        """Get current user profile"""
        return self.current_user_profile

    def update_user_profile(self, profile: UserProfile):
        """Update user profile in database"""
        if not profile.name:
            return
            
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Update user's todo counter
                cursor.execute("""
                    UPDATE users SET todo_counter = ? WHERE name = ?
                """, (profile._todo_counter, profile.name))
                
                conn.commit()
                
        self.current_user_profile = profile

    def add_todo(self, task: str) -> Todo:
        """Add a new todo for current user"""
        if not self.current_user_name:
            raise Exception("No user selected")
            
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get and increment todo counter
                cursor.execute("""
                    UPDATE users SET todo_counter = todo_counter + 1 WHERE name = ?
                """, (self.current_user_name,))
                
                cursor.execute("""
                    SELECT todo_counter FROM users WHERE name = ?
                """, (self.current_user_name,))
                
                todo_counter = cursor.fetchone()['todo_counter']
                
                # Create the todo
                created_at = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO todos (user_name, user_todo_id, task, completed, created_at)
                    VALUES (?, ?, ?, 0, ?)
                """, (self.current_user_name, todo_counter, task, created_at))
                
                conn.commit()
                
                # Create todo object
                todo = Todo(
                    id=todo_counter,
                    task=task,
                    completed=False,
                    created_at=created_at
                )
                
                # Update current profile
                self.current_user_profile.todos.append(todo)
                self.current_user_profile._todo_counter = todo_counter
                
                return todo

    def remove_todo(self, task_reference: str) -> bool:
        """Mark a todo as completed"""
        if not self.current_user_name:
            return False
            
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Try to find by task content first
                cursor.execute("""
                    SELECT user_todo_id FROM todos 
                    WHERE user_name = ? AND completed = 0 AND LOWER(task) LIKE LOWER(?)
                    LIMIT 1
                """, (self.current_user_name, f"%{task_reference}%"))
                
                result = cursor.fetchone()
                todo_id = None
                
                if result:
                    todo_id = result['user_todo_id']
                else:
                    # Try to parse as ID
                    try:
                        todo_id = int(task_reference)
                    except ValueError:
                        return False
                
                if todo_id:
                    # Mark as completed
                    completed_at = datetime.now().isoformat()
                    cursor.execute("""
                        UPDATE todos 
                        SET completed = 1, completed_at = ? 
                        WHERE user_name = ? AND user_todo_id = ? AND completed = 0
                    """, (completed_at, self.current_user_name, todo_id))
                    
                    if cursor.rowcount > 0:
                        conn.commit()
                        
                        # Update current profile
                        for todo in self.current_user_profile.todos:
                            if todo.id == todo_id and not todo.completed:
                                todo.completed = True
                                todo.completed_at = completed_at
                                break
                        
                        return True
                
                return False

    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        if not self.current_user_name:
            return
            
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                timestamp = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO conversation_history (user_name, role, content, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (self.current_user_name, role, content, timestamp))
                
                # Keep only last 50 messages per user
                cursor.execute("""
                    DELETE FROM conversation_history 
                    WHERE user_name = ? AND id NOT IN (
                        SELECT id FROM conversation_history 
                        WHERE user_name = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 50
                    )
                """, (self.current_user_name, self.current_user_name))
                
                conn.commit()
                
                # Update current profile
                msg = Message(role=role, content=content, timestamp=timestamp)
                self.current_user_profile.conversation_history.append(msg)
                
                # Keep only last 50 in memory
                if len(self.current_user_profile.conversation_history) > 50:
                    self.current_user_profile.conversation_history = self.current_user_profile.conversation_history[-50:]

    def clear_name_only(self):
        """Clear current user name (for name change)"""
        with self._lock:
            if self.current_user_profile:
                self.current_user_profile.name = None
            self.current_user_name = None

    def is_available(self) -> bool:
        """Check if database is available"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            print(f"Database availability check failed: {e}")
            return False

    def get_user_stats(self) -> Dict[str, Any]:
        """Get statistics about users and todos"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get user count
                cursor.execute("SELECT COUNT(*) as count FROM users")
                user_count = cursor.fetchone()['count']
                
                # Get total todos
                cursor.execute("SELECT COUNT(*) as count FROM todos")
                total_todos = cursor.fetchone()['count']
                
                # Get completed todos
                cursor.execute("SELECT COUNT(*) as count FROM todos WHERE completed = 1")
                completed_todos = cursor.fetchone()['count']
                
                return {
                    'user_count': user_count,
                    'total_todos': total_todos,
                    'completed_todos': completed_todos,
                    'active_todos': total_todos - completed_todos
                }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}


# LangChain Tool Classes
class AddTodoTool(BaseTool):
    name = "add_todo"
    description = "Add a new todo item to the user's list"
    memory_store: Any = Field(...)

    def _run(self, task: str) -> str:
        """Add a todo item"""
        try:
            todo = self.memory_store.add_todo(task)
            return f"✅ Added '{task}' to your todo list!"
        except Exception as e:
            return f"❌ Error adding todo: {str(e)}"

    async def _arun(self, task: str) -> str:
        """Async version"""
        return self._run(task)


class RemoveTodoTool(BaseTool):
    name = "remove_todo"
    description = "Remove or complete a todo item from the user's list"
    memory_store: Any = Field(...)

    def _run(self, task_reference: str) -> str:
        """Remove a todo item"""
        try:
            if self.memory_store.remove_todo(task_reference):
                return f"✅ Completed and removed '{task_reference}' from your todo list!"
            else:
                return f"❌ Couldn't find a todo matching '{task_reference}'. Use 'show todos' to see your current tasks."
        except Exception as e:
            return f"❌ Error removing todo: {str(e)}"

    async def _arun(self, task_reference: str) -> str:
        """Async version"""
        return self._run(task_reference)


class ShowTodosTool(BaseTool):
    name = "show_todos"
    description = "Show all active todo items"
    memory_store: Any = Field(...)

    def _run(self) -> str:
        """Show all active todos"""
        try:
            profile = self.memory_store.get_user_profile()
            active_todos = [todo for todo in profile.todos if not todo.completed]
            
            if not active_todos:
                return "📝 You don't have any active todos right now."
            
            todo_list = "\n".join([f"{i+1}. {todo.task}" for i, todo in enumerate(active_todos)])
            return f"📝 Here are your active todos:\n{todo_list}"
        except Exception as e:
            return f"❌ Error getting todos: {str(e)}"

    async def _arun(self) -> str:
        """Async version"""
        return self._run()


class ShowStatsTool(BaseTool):
    name = "show_stats"
    description = "Show user statistics and todo counts"
    memory_store: Any = Field(...)

    def _run(self) -> str:
        """Show user statistics"""
        try:
            profile = self.memory_store.get_user_profile()
            active_count = len([todo for todo in profile.todos if not todo.completed])
            completed_count = len([todo for todo in profile.todos if todo.completed])
            
            return f"📊 Your stats:\n• Active todos: {active_count}\n• Completed todos: {completed_count}\n• Total todos: {active_count + completed_count}"
        except Exception as e:
            return f"❌ Error getting stats: {str(e)}"

    async def _arun(self) -> str:
        """Async version"""
        return self._run()


# LangGraph State Definition
class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_profile: Optional[UserProfile]
    current_task: Optional[str]
    tool_result: Optional[str]


class LangGraphChatbot:
    def __init__(self, memory_store):
        if not memory_store.is_available():
            raise Exception("Memory store not available for chatbot initialization")
        
        self.memory_store = memory_store
        
        # Initialize LangChain LLM
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=os.getenv('GEMINI_API_KEY'),
                temperature=0.3,
                convert_system_message_to_human=True
            )
            print("DEBUG: LangChain Gemini LLM initialized successfully")
        except Exception as e:
            print(f"DEBUG: LangChain LLM initialization failed: {e}")
            self.llm = None
        
        # Initialize tools
        self.tools = {
            "add_todo": AddTodoTool(memory_store=memory_store),
            "remove_todo": RemoveTodoTool(memory_store=memory_store),
            "show_todos": ShowTodosTool(memory_store=memory_store),
            "show_stats": ShowStatsTool(memory_store=memory_store)
        }
        
        # Build LangGraph workflow
        self.workflow = self._build_graph()
        
        print("DEBUG: LangGraph Chatbot initialized successfully")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def chatbot_node(state: AgentState) -> AgentState:
            """Main chatbot node that processes messages"""
            messages = state.get("messages", [])
            user_profile = state.get("user_profile")
            
            if not messages:
                return state
            
            last_message = messages[-1]
            if not isinstance(last_message, HumanMessage):
                return state
            
            user_input = last_message.content
            
            # Intent detection
            intent_result = self._detect_intent(user_input, user_profile)
            
            if intent_result["action"] in self.tools:
                # Execute tool
                tool_response = self._execute_tool(intent_result["action"], intent_result["parameters"])
                response_content = tool_response
            else:
                # Use LLM for general chat
                response_content = self._generate_llm_response(user_input, user_profile)
            
            # Add AI response to messages
            ai_message = AIMessage(content=response_content)
            updated_messages = messages + [ai_message]
            
            return {
                **state,
                "messages": updated_messages,
                "tool_result": response_content
            }
        
        # Create workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("chatbot", chatbot_node)
        workflow.set_entry_point("chatbot")
        workflow.add_edge("chatbot", END)
        
        return workflow.compile()

    def _detect_intent(self, user_input: str, user_profile: UserProfile) -> Dict[str, Any]:
        """Detect user intent using LLM"""
        
        # Format context
        active_todos = [todo for todo in user_profile.todos if not todo.completed] if user_profile else []
        todos_context = "\n".join([f"- {todo.id}: {todo.task}" for todo in active_todos]) if active_todos else "No active todos"
        
        system_prompt = f"""
You are an intelligent assistant that analyzes user messages to detect their intent and extract structured data.

Available actions:
- "add_todo": Add a task. Extract: "task"
- "remove_todo": Remove or mark a task done. Extract: "task_reference"
- "show_todos": Show active todos
- "show_stats": Show user statistics
- "chat": General conversation
- "name_change": Change user name

User context:
- Name: {user_profile.name if user_profile else 'Unknown'}
- Todos:
{todos_context}

Respond ONLY in this format:
{{
  "action": "action_name",
  "parameters": {{
    "task": "value",           // for add_todo
    "task_reference": "value"  // for remove_todo
  }},
  "confidence": 0.95
}}

Examples:
- "Add buy milk" → {{"action": "add_todo", "parameters": {{"task": "buy milk"}}, "confidence": 0.95}}
- "Remove buy milk" → {{"action": "remove_todo", "parameters": {{"task_reference": "buy milk"}}, "confidence": 0.95}}
- "Show my todos" → {{"action": "show_todos", "parameters": {{}}, "confidence": 0.95}}
- "What's up?" → {{"action": "chat", "parameters": {{}}, "confidence": 0.95}}

User: "{user_input}"
"""

        
        try:
            if self.llm:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_input)
                ]
                response = self.llm.invoke(messages)
                
                # Extract JSON from response
                response_text = response.content.strip()
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_text = response_text[json_start:json_end]
                    return json.loads(json_text)
                else:
                    return json.loads(response_text)
            else:
                # Fallback intent detection
                return self._fallback_intent_detection(user_input)
                
        except Exception as e:
            print(f"DEBUG: Intent detection error: {e}")
            return self._fallback_intent_detection(user_input)

    def _fallback_intent_detection(self, user_input: str) -> Dict[str, Any]:
        """Fallback intent detection using keyword matching"""
        message_lower = user_input.lower().strip()
        
        if any(word in message_lower for word in ['add', 'create', 'new']) and any(word in message_lower for word in ['todo', 'task']):
            task = self._extract_task_basic(user_input)
            return {
                "action": "add_todo",
                "parameters": {"task": task},
                "confidence": 0.6
            }
        
        elif any(word in message_lower for word in ['remove', 'delete', 'complete', 'done']):
            task_ref = self._extract_task_reference_basic(user_input)
            return {
                "action": "remove_todo",
                "parameters": {"task_reference": task_ref},
                "confidence": 0.6
            }
        
        elif any(word in message_lower for word in ['show', 'list', 'display']) and any(word in message_lower for word in ['todo', 'task']):
            return {
                "action": "show_todos",
                "parameters": {},
                "confidence": 0.7
            }
        
        elif any(word in message_lower for word in ['stats', 'statistics', 'count']):
            return {
                "action": "show_stats",
                "parameters": {},
                "confidence": 0.7
            }
        
        elif any(word in message_lower for word in ['change', 'update']) and 'name' in message_lower:
            return {
                "action": "name_change",
                "parameters": {},
                "confidence": 0.8
            }
        
        return {
            "action": "chat",
            "parameters": {},
            "confidence": 0.4
        }

    def _extract_task_basic(self, message: str) -> str:
        """Basic task extraction"""
        message_lower = message.lower()
        for word in ['add', 'create', 'new', 'todo', 'task', 'to', 'my', 'list']:
            message_lower = message_lower.replace(word, '')
        return message_lower.strip() or "new task"

    def _extract_task_reference_basic(self, message: str) -> str:
        """Basic task reference extraction"""
        message_lower = message.lower()
        for word in ['remove', 'delete', 'complete', 'done', 'todo', 'task', 'the']:
            message_lower = message_lower.replace(word, '')
        return message_lower.strip() or "1"

    def _execute_tool(self, action: str, parameters: Dict[str, Any]) -> str:
        """Execute a tool based on action and parameters"""
        try:
            tool = self.tools.get(action)
            if not tool:
                return f"❌ Unknown action: {action}"
            
            if action == "add_todo":
                task = parameters.get("task", "").strip()
                if not task:
                    return "❌ Please specify a task to add."
                return tool._run(task)
            
            elif action == "remove_todo":
                task_reference = parameters.get("task_reference", "").strip()
                if not task_reference:
                    return "❌ Please specify which task to remove."
                return tool._run(task_reference)
            
            elif action in ["show_todos", "show_stats"]:
                return tool._run()
            
            else:
                return f"❌ Action {action} not implemented"
                
        except Exception as e:
            return f"❌ Error executing {action}: {str(e)}"

    def _generate_llm_response(self, user_input: str, user_profile: UserProfile) -> str:
        """Generate conversational response using LLM"""
        try:
            if not self.llm:
                return "Hello! I'm here to help you manage your todos. You can add tasks, remove tasks, or show your current list."
            
            # Build context
            active_todos = [todo for todo in user_profile.todos if not todo.completed] if user_profile else []
            todos_context = f"User has {len(active_todos)} active todos." if active_todos else "User has no active todos."
            
            system_prompt = f"""You are a helpful personal assistant. Be friendly and conversational.

USER CONTEXT:
- User name: {user_profile.name if user_profile else 'Unknown'}
- {todos_context}

Keep responses concise and helpful. If the user is asking about todos, guide them to use specific commands like:
- "add [task]" to add a new todo
- "remove [task or number]" to complete a todo
- "show todos" to see current tasks
- "show stats" to see statistics

Respond naturally and be encouraging."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"DEBUG: LLM response generation error: {e}")
            try:
                # Minimal fallback attempt with just user message
                if self.llm:
                    messages = [
                        SystemMessage(content="You are a helpful assistant. Respond naturally and clearly."),
                        HumanMessage(content=user_input)
                    ]
                    response = self.llm.invoke(messages)
                    return response.content
                else:
                    return "Sorry, I'm currently unavailable."
            except Exception as fallback_error:
                print(f"DEBUG: Fallback LLM error: {fallback_error}")
                return "Sorry, I'm currently unable to generate a response."


    def chat(self, message: str) -> str:
        """Main chat method using LangGraph workflow"""
        try:
            # Add user message to history
            self.memory_store.add_message("user", message)
            
            # Get user profile
            user_profile = self.memory_store.get_user_profile()
            
            # Check for name change request
            if 'change' in message.lower() and 'name' in message.lower():
                return "I'll help you change your name. Redirecting you to set a new name..."
            
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "user_profile": user_profile,
                "current_task": None,
                "tool_result": None
            }
            
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            # Extract response
            response = result.get("tool_result", "I'm sorry, I couldn't process your request.")
            
            # Add assistant response to history
            self.memory_store.add_message("assistant", response)
            
            return response
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"DEBUG: Chat error: {str(e)}")
            return error_msg


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global variables
memory_store = None
chatbot = None

# Initialize services on startup
try:
    memory_store = SQLiteMemoryStore()
    chatbot = LangGraphChatbot(memory_store)
except Exception as e:
    print(f"CRITICAL ERROR: Could not initialize services: {str(e)}")


@app.before_request
def ensure_services():
    global memory_store, chatbot

    if memory_store is None or not memory_store.is_available():
        try:
            memory_store = SQLiteMemoryStore()
            chatbot = LangGraphChatbot(memory_store)
        except Exception as e:
            return jsonify({'error': 'Database not available. Please try again.'}), 503

    if 'user_name' in session:
        memory_store.switch_user(session['user_name'])


@app.route('/')
def index():
    """Main route that checks if user has provided name"""
    try:
        profile = memory_store.get_user_profile()
        
        print(f"DEBUG: Profile name: {profile.name}")
        print(f"DEBUG: Has name: {profile.name is not None}")
        
        if profile.name:
            try:
                with open('templates/index.html', 'r') as f:
                    return f.read()
            except FileNotFoundError:
                return "<h1>Chat Interface</h1><p>Templates not found. Please ensure templates/index.html exists.</p>"
        else:
            try:
                with open('templates/welcome.html', 'r') as f:
                    return f.read()
            except FileNotFoundError:
                return "<h1>Welcome</h1><p>Templates not found. Please ensure templates/welcome.html exists.</p>"
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500


@app.route('/welcome')
def welcome():
    """Welcome page for new users"""
    try:
        with open('templates/welcome.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Welcome</h1><p>Please enter your name to continue.</p>"

@app.route('/set_name', methods=['POST'])
def set_name():
    """Set user's name and redirect to chat"""
    try:
        name = request.json.get('name', '').strip()
        if not name:
            return jsonify({'error': 'Name cannot be empty'}), 400
        
        print(f"DEBUG: Setting name to: {name}")
        
        result = memory_store.switch_user(name)
        session['user_name'] = name
        
        profile_after = memory_store.get_user_profile()
        print(f"DEBUG: Profile after update: {profile_after.name}")
        
        return jsonify({
            'message': result,
            'success': True
        })
    except Exception as e:
        print(f"DEBUG: Error in set_name: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Chat API endpoint - Now powered by LangChain and LangGraph"""
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if user has name set
        profile = memory_store.get_user_profile()
        if not profile.name:
            return jsonify({'error': 'Please set your name first'}), 400
        
        # Check for name change request
        if 'change' in user_message.lower() and 'name' in user_message.lower():
            memory_store.clear_name_only()
            session.pop('user_name', None)
            
            return jsonify({
                'response': "I'll help you change your name. Redirecting you to set a new name...",
                'redirect_to_welcome': True,
                'success': True
            })
        
        # Process message with LangGraph-powered chatbot
        response = chatbot.chat(user_message)
        
        return jsonify({
            'response': response,
            'user_name': profile.name,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"DEBUG: Chat API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/change_name', methods=['POST'])
def change_name():
    """Clear current name and prepare for name change"""
    try:
        profile = memory_store.get_user_profile()
        if not profile.name:
            return jsonify({'error': 'No name to change'}), 400
        
        memory_store.clear_name_only()
        session.pop('user_name', None)
        
        return jsonify({
            'message': 'Name cleared. You can now set a new name.',
            'success': True,
            'redirect_to_welcome': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/todos', methods=['GET'])
def get_todos():
    """Get all active todos"""
    try:
        profile = memory_store.get_user_profile()
        if not profile.name:
            return jsonify({'error': 'Please set your name first'}), 400
        
        active_todos = [
            {
                'id': todo.id,
                'task': todo.task,
                'created_at': todo.created_at
            }
            for todo in profile.todos if not todo.completed
        ]
        return jsonify({
            'todos': active_todos,
            'count': len(active_todos)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/todos/<int:todo_id>', methods=['DELETE'])
def delete_todo(todo_id):
    """Delete a specific todo"""
    try:
        profile = memory_store.get_user_profile()
        if not profile.name:
            return jsonify({'error': 'Please set your name first'}), 400
        
        for todo in profile.todos:
            if todo.id == todo_id and not todo.completed:
                todo.completed = True
                todo.completed_at = datetime.now().isoformat()
                memory_store.update_user_profile(profile)
                return jsonify({
                    'message': f"✅ Completed '{todo.task}' and removed it from your to-do list!",
                    'success': True
                })
        
        return jsonify({'error': 'Todo not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/todos', methods=['POST'])
def add_todo():
    """Add a new todo"""
    try:
        task = request.json.get('task', '').strip()
        if not task:
            return jsonify({'error': 'Task cannot be empty'}), 400
        
        profile = memory_store.get_user_profile()
        if not profile.name:
            return jsonify({'error': 'Please set your name first'}), 400
        
        result = chatbot.todo_tools.add_todo_tool.invoke({"task": task})
        
        return jsonify({
            'message': result,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile', methods=['GET'])
def get_profile():
    """Get user profile"""
    try:
        profile = memory_store.get_user_profile()
        return jsonify({
            'name': profile.name,
            'todo_count': len([todo for todo in profile.todos if not todo.completed]),
            'completed_count': len([todo for todo in profile.todos if todo.completed]),
            'has_name': profile.name is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile', methods=['PUT'])
def update_profile():
    """Update user profile"""
    try:
        name = request.json.get('name', '').strip()
        if not name:
            return jsonify({'error': 'Name cannot be empty'}), 400
        
        result = memory_store.switch_user(name)
        session['user_name'] = name
        
        return jsonify({
            'message': result,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation_history', methods=['GET'])
def get_conversation_history():
    """Get recent conversation history"""
    try:
        profile = memory_store.get_user_profile()
        if not profile.name:
            return jsonify({'error': 'Please set your name first'}), 400
        
        recent_messages = profile.conversation_history[-10:] if profile.conversation_history else []
        
        return jsonify({
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp
                }
                for msg in recent_messages
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_data():
    """Reset user data (for testing purposes)"""
    try:
        global memory_store, chatbot
        memory_store = SQLiteMemoryStore()
        chatbot = LangGraphChatbot(memory_store)
        
        session.clear()
        
        return jsonify({
            'message': 'All data has been reset',
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        is_healthy = True
        errors = []
        
        if memory_store is None:
            is_healthy = False
            errors.append("Memory store is None")
        elif not memory_store.is_available():
            is_healthy = False
            errors.append("Memory store not available")
        
        if chatbot is None:
            is_healthy = False
            errors.append("Chatbot is None")
        
        profile = None
        if memory_store and memory_store.is_available():
            try:
                profile = memory_store.get_user_profile()
            except Exception as e:
                is_healthy = False
                errors.append(f"Profile access error: {str(e)}")
        
        return jsonify({
            'status': 'healthy' if is_healthy else 'unhealthy',
            'errors': errors,
            'chatbot_loaded': chatbot is not None,
            'memory_store_loaded': memory_store is not None,
            'memory_store_available': memory_store.is_available() if memory_store else False,
            'user_has_name': profile.name is not None if profile else False,
            'active_todos': len([todo for todo in profile.todos if not todo.completed]) if profile else 0,
            'langchain_enabled': True,
            'langgraph_enabled': True,
            'gemini_enabled': True
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)