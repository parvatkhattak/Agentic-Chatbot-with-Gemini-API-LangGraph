# 🤖 Agentic Chatbot with Gemini API & LangGraph

A sophisticated conversational AI assistant built with Google's Gemini API and LangGraph that provides intelligent conversation management, persistent memory, and productivity tools through an advanced agentic architecture. Features both web-based and CLI interfaces for maximum flexibility.

## 🌟 **Key Features & Highlights**

### 🧠 **Advanced AI Architecture**
- **Agentic Design**: Uses LangGraph for sophisticated workflow management
- **Intent Detection**: Gemini-powered natural language understanding
- **Multi-Tool Integration**: Seamlessly executes tools based on user intent
- **Context Awareness**: Maintains conversation history and user context
- **Fallback Intelligence**: Graceful handling of edge cases and errors

### 💾 **Sophisticated Memory System**
- **Multi-User Support**: Complete user isolation with individual profiles
- **Persistent Storage**: SQLite database with automatic schema management
- **Conversation History**: Maintains context across sessions (last 50 messages)
- **User Switching**: Seamless profile switching with data preservation
- **Memory Optimization**: Automatic cleanup and efficient data management
- **Transaction Safety**: ACID-compliant database operations

### 🔧 **Intelligent Tool System**
- **Smart Todo Management**: Natural language task manipulation
- **User Profile Management**: Dynamic name setting and user switching
- **Conversation Tools**: History access and context management
- **Flexible Tool Execution**: Intent-based tool selection and parameter extraction

## 🏗️ **Architecture Overview**

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   Gemini Intent │───▶│   LangGraph     │
│   (CLI/Web)     │    │   Detection     │    │   Workflow      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                              ┌─────────────────┐    ┌─────────────────┐
                              │   Tool System   │────│ SQLite Database │
                              │   Execution     │    │  (Multi-User)   │
                              └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. **LangGraph Workflow Engine**
```python
def _build_graph(self) -> StateGraph:
    """Build the LangGraph workflow"""
    
    def chatbot_node(state: AgentState) -> AgentState:
        """Main chatbot reasoning node with intent detection"""
        # Intent detection with Gemini
        intent, extracted_info = self.intent_detector.detect_intent(user_message, current_user)
        
        # Dynamic tool execution based on intent
        if intent in self.tools and extracted_info:
            result = self.tools[intent]._run(extracted_info)
        else:
            # Fallback to general conversation
            result = self.llm.invoke(messages)
```

#### 2. **Advanced Intent Detection**
```python
class GeminiIntentDetector:
    def detect_intent(self, text: str, current_user: Optional[str] = None) -> tuple[str, str]:
        """Use Gemini to detect intent and extract relevant information"""
        
        system_prompt = """You are an intent detection system for a personal assistant chatbot.
        
        Available intents:
        - "add_todo": User wants to add a task to their to-do list
        - "list_todos": User wants to see their current to-do list
        - "remove_todo": User wants to remove/complete a task
        - "update_name": User wants to set/change their name
        - "chat": General conversation or questions
        
        Return JSON: {"intent": "intent_name", "extracted_info": "relevant_info"}
        """
```

#### 3. **SQLite Multi-User Memory Architecture**
```python
class SQLiteMemoryStore:
    def __init__(self, db_path="webapp_memory.db"):
        self.db_path = db_path
        self.current_user_profile = UserProfile()
        self.current_user_name = None
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        # Creates users, todos, and conversation_history tables
        # Includes proper indexes for optimal performance
        # Handles foreign key relationships
    
    def switch_user(self, name: str) -> str:
        """Switch to a specific user, creating new profile if needed"""
        # Case-insensitive user matching
        # Automatic profile creation
        # Data preservation across sessions
```

## 🚀 **Getting Started**

### Prerequisites
- Python 3.8+
- Google Gemini API key
- Required Python packages (see installation)

### Installation

#### **Quick Setup**
```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-chatbot.git
cd agentic-chatbot

# Install dependencies
pip install -r requirements.txt

# Set up your Google API key
export GEMINI_API_KEY="your_gemini_api_key_here"
```

#### **Dependencies**
```bash
# Core dependencies
pip install google-generativeai langchain langchain-google-genai langgraph pydantic flask sqlite3
```

### Configuration
Create a `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

## 💬 **Usage Examples**

### **CLI Interface**
```bash
# Start the CLI chatbot
python chatbot.py

# Interactive session
🤖 Welcome to your Personal Assistant Chatbot!
👋 Welcome! What's your name?

You: My name is Alex
Chatbot: 👋 Nice to meet you, Alex! I've created a new profile for you.
📝 Your to-do list is empty.

You: Add "learn LangGraph" to my todo list
Chatbot: ✅ Added 'learn LangGraph' to your to-do list.

You: What's on my todo list?
Chatbot: 📋 Here's your current to-do list:
1. learn LangGraph (added 07/06)

You: I finished learning LangGraph
Chatbot: ✅ Completed 'learn LangGraph' and removed it from your to-do list!
```

### **Web Interface**
```bash
# Start the web application
python web.py

# Access at http://localhost:5000
```

## 🔧 **Advanced Features**

### **1. Natural Language Processing**
- **Flexible Intent Recognition**: Handles various ways of expressing the same intent
- **Context Awareness**: Understands references to previous conversations
- **Parameter Extraction**: Automatically extracts relevant information from user input

```python
# Examples of natural language flexibility:
"Add groceries to my list" → add_todo: "groceries"
"I need to buy milk" → add_todo: "buy milk"
"Put down learn Python" → add_todo: "learn Python"
"I finished the groceries task" → remove_todo: "groceries"
"Done with milk" → remove_todo: "milk"
```

### **2. Multi-User Management**
- **User Isolation**: Complete data separation between users
- **Profile Switching**: Seamless switching between user profiles
- **Data Persistence**: Maintains user data across sessions

```python
# User management examples:
You: Change my name to Sarah
Chatbot: 👋 Nice to meet you, Sarah! I've created a new profile for you.

You: Switch to Alex
Chatbot: 👋 Welcome back, Alex! I've loaded your previous data.
```

### **3. Intelligent Tool System**
```python
# Available tools with smart execution:
tools = {
    "add_todo": AddTodoTool(),
    "show_todos": ShowTodosTool(),
    "remove_todo": RemoveTodoTool(),
    "show_stats": ShowStatsTool()
}
```

## 📊 **Data Models**

### **Database Schema**
```sql
-- Users table
CREATE TABLE users (
    name TEXT PRIMARY KEY,
    todo_counter INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Todos table
CREATE TABLE todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT NOT NULL,
    user_todo_id INTEGER NOT NULL,
    task TEXT NOT NULL,
    completed BOOLEAN DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    FOREIGN KEY (user_name) REFERENCES users (name),
    UNIQUE(user_name, user_todo_id)
);

-- Conversation history table
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_name) REFERENCES users (name)
);
```

### **User Profile Structure**
```python
class UserProfile(BaseModel):
    name: Optional[str] = None
    todos: List[Todo] = []
    todo_counter: int = 0
    conversation_history: List[Message] = []

class Todo(BaseModel):
    id: int
    task: str
    created_at: str
    completed: bool = False
    completed_at: Optional[str] = None
```

### **Memory Storage**
- **Persistent Storage**: SQLite database with ACID compliance
- **Multi-User Support**: Separate data for each user with foreign key relationships
- **Conversation History**: Last 50 messages per user with automatic cleanup
- **Error Recovery**: Graceful handling of database connection issues
- **Thread Safety**: Proper locking mechanisms for concurrent access

## 🔍 **Technical Implementation**

### **Database Features**
- **ACID Compliance**: Ensures data integrity with transaction support
- **Optimized Queries**: Proper indexing for fast data retrieval
- **Connection Pooling**: Efficient database connection management
- **Backup Ready**: Easy database backup and migration
- **Concurrent Access**: Thread-safe operations with proper locking

### **LangGraph State Management**
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_profile: Optional[UserProfile]
    current_task: Optional[str]
    tool_result: Optional[str]
```

### **Conversation Flow**
1. **User Input**: Message received via CLI or web interface
2. **Intent Detection**: Gemini API analyzes user intent
3. **Tool Execution**: Appropriate tool selected and executed
4. **Response Generation**: Context-aware response created
5. **Memory Update**: Conversation history and user data updated in SQLite

### **Error Handling**
- **Graceful Degradation**: Fallback to chat mode if tools fail
- **Input Validation**: Comprehensive parameter validation
- **Database Recovery**: Automatic recovery from connection issues
- **API Resilience**: Handles API failures gracefully
- **Transaction Rollback**: Proper error handling with database rollbacks

## 🎯 **Example Interactions**

### **Todo Management**
```
You: "I need to remember to call mom"
AI: "✅ Added 'call mom' to your to-do list."

You: "What do I need to do today?"
AI: "📋 Here's your current to-do list:
1. call mom (added 07/06)
2. finish project (added 07/05)"

You: "I called mom"
AI: "✅ Completed 'call mom' and removed it from your to-do list!"
```

### **User Management**
```
You: "My name is John"
AI: "👋 Nice to meet you, John! I've created a new profile for you."

You: "Switch to Sarah"
AI: "👋 Welcome back, Sarah! I've loaded your previous data."

You: "Show my stats"
AI: "📊 Your stats:
• Active todos: 3
• Completed todos: 7
• Total todos: 10"
```

### **General Conversation**
```
You: "How does this chatbot work?"
AI: "I'm an AI assistant that can help you with conversations and manage your to-do list. I use Google's Gemini API to understand your intent and can:
- Have natural conversations while remembering context
- Manage personal to-do lists for multiple users
- Switch between different user profiles
- Remember names and past conversations for each user"
```

## 🛠️ **Development Setup**

### **Development Environment**
```bash
# Set up development environment
pip install -r requirements-dev.txt

# Run with debug mode
export DEBUG=True
python web.py
```

### **Database Management**
```bash
# View database contents
sqlite3 webapp_memory.db

# Backup database
cp webapp_memory.db webapp_memory_backup.db

# Reset database (for testing)
rm webapp_memory.db
python web.py  # Will recreate database automatically
```

## 📈 **Performance & Scalability**

### **Optimization Features**
- **Database Indexing**: Optimized queries with proper indexes
- **Connection Management**: Efficient SQLite connection handling
- **Memory Optimization**: Automatic cleanup of old conversations
- **Smart API Usage**: Optimized Gemini API calls
- **Fast User Switching**: Efficient database queries for user profiles

### **Scalability Considerations**
- **Database Migration**: Easy transition to PostgreSQL or other databases
- **Microservices Ready**: Modular architecture for easy scaling
- **Load Balancing**: Stateless design for horizontal scaling
- **Caching**: Database-level caching for frequently accessed data
- **Concurrent Users**: Thread-safe operations for multiple simultaneous users

## 🔒 **Security & Privacy**

### **Data Protection**
- **Local Storage**: All data stored locally in SQLite database
- **User Isolation**: Complete data separation between users via foreign keys
- **Input Sanitization**: Prevents SQL injection attacks
- **API Key Security**: Environment-based configuration
- **Transaction Safety**: ACID compliance ensures data integrity

### **Privacy Features**
- **No Data Sharing**: Conversations remain private and local
- **User Control**: Users can reset their data anytime
- **Conversation History**: Limited to last 50 messages per user
- **Data Encryption**: Optional database encryption support

## 🚀 **Advanced Use Cases**

### **Personal Productivity**
- **Task Management**: Smart todo list with natural language input
- **Goal Tracking**: Track completed tasks and progress with statistics
- **Context Switching**: Seamless switching between personal and work profiles

### **Multi-User Scenarios**
- **Family Use**: Each family member has their own profile
- **Team Collaboration**: Shared device with individual user data
- **Development**: Multiple user profiles for testing

### **Integration Possibilities**
- **Calendar Integration**: Sync with Google Calendar
- **Notification System**: Remind users of pending tasks
- **Voice Interface**: Add speech-to-text capabilities
- **Mobile App**: Native mobile applications with SQLite sync

## 🔮 **Future Enhancements**

### **Short-term Roadmap**
- **Enhanced Web Interface**: Complete Flask web application with modern UI
- **Rich Media Support**: Handle images and files in conversations
- **Advanced Search**: Full-text search across conversations and todos
- **Export Features**: Export todo lists and conversations to various formats
- **Database Migrations**: Automated schema updates

### **Long-term Vision**
- **Plugin Architecture**: Extensible tool system with custom plugins
- **AI Agents**: Multi-agent collaboration with specialized agents
- **Enterprise Features**: Team management and admin tools
- **Cloud Sync**: Optional cloud database synchronization
- **Real-time Notifications**: WebSocket-based real-time updates

## 🤝 **Contributing**

### **Development Guidelines**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### **Code Style**
- Follow PEP 8 guidelines
- Use type hints for all functions
- Include comprehensive docstrings
- Add unit tests for new features
- Test database migrations thoroughly

### **Database Guidelines**
- Always use transactions for multiple operations
- Include proper error handling for database operations
- Test with multiple concurrent users
- Ensure backward compatibility for schema changes

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Google AI Studio** for the powerful Gemini API
- **LangChain Team** for the excellent LangGraph framework
- **SQLite Team** for the robust embedded database
- **Open Source Community** for inspiration and best practices

---

## 🎯 **Why This Project Stands Out**

### **Technical Excellence**
- **Modern Architecture**: Uses cutting-edge LangGraph for workflow management
- **Advanced AI Integration**: Sophisticated intent detection with Gemini API
- **Production Ready**: Comprehensive error handling and data persistence
- **Scalable Design**: Modular architecture with robust database foundation

### **User Experience**
- **Natural Interaction**: Handles various ways of expressing the same intent
- **Context Awareness**: Remembers conversations and user preferences
- **Multi-User Support**: Complete user isolation and profile management
- **Flexible Interface**: Both CLI and web interface support

### **Innovation**
- **Agentic Architecture**: Demonstrates advanced AI agent patterns
- **Intelligent Tool Use**: Dynamic tool selection based on user intent
- **Memory Management**: Sophisticated multi-user memory system with SQLite
- **Error Resilience**: Graceful handling of edge cases and failures
- **Database Integration**: Professional-grade data persistence and management

This project showcases a complete understanding of modern AI agent architecture, combining the power of large language models with structured workflow management and robust database storage to create a truly intelligent and reliable assistant.