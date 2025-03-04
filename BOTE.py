from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_community.tools import TavilySearchResults
import datetime
from langgraph.graph import StateGraph
from langchain.agents import initialize_agent

# Load environment variables
load_dotenv()

# Initialize the LLMs (Google Gemini Models)
gemini_llm_1 = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
gemini_llm_2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format."""
    current_time = datetime.datetime.now()
    return current_time.strftime(format)

# Define tools for the agents
search_tool = TavilySearchResults(search_depth="basic")

agent1_tools = [
    Tool(name="SearchTool", func=search_tool.invoke, description="Searches the web for the latest information."),
]

agent2_tools = [
    Tool(name="SystemTime", func=get_system_time, description="Gets the current system time."),
]

# Create LangChain agents
agent1 = initialize_agent(tools=agent1_tools, llm=gemini_llm_1, agent="zero-shot-react-description", verbose=True)
agent2 = initialize_agent(tools=agent2_tools, llm=gemini_llm_2, agent="zero-shot-react-description", verbose=True)

graph = StateGraph(dict)
graph.add_node("agent1", agent1.invoke)
graph.add_node("agent2", agent2.invoke)

graph.add_edge("agent1", "agent2")
graph.set_entry_point("agent1")

graph = graph.compile()

# Function to interact with the chatbot
def chat_with_bot(user_input):
    response = graph.invoke({"input": user_input})
    return response["output"]

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        bot_response = chat_with_bot(user_input)
        print("Bot:", bot_response)
