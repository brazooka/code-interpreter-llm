from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.tools import TavilySearchResults


load_dotenv()


def setup_agents() -> AgentExecutor:
    """Setup and return the grand agent that can use both Python and CSV agents."""
    
    # Setup Python Agent
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    You have qrcode package installed
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    # Setup Python Agent
    python_agent = create_react_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=[PythonREPLTool()],
        prompt=prompt
    )
    python_agent_executor = AgentExecutor(
        agent=python_agent,
        tools=[PythonREPLTool()],
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    # Setup CSV Agent
    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )

    # Setup Weather Agent
    weather_agent = create_react_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=[TavilySearchResults()],
        prompt=base_prompt.partial(instructions="""You are a weather information agent.
        Use the search tool to find current weather information and forecasts.
        Always try to provide accurate and up-to-date weather information.
        If you can't find specific weather data, explain what information you can provide.""")
    )
    weather_agent_executor = AgentExecutor(
        agent=weather_agent,
        tools=[TavilySearchResults()],
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    # Setup Grand Agent
    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    def weather_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return weather_agent_executor.invoke({"input": original_prompt})

    grand_tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                         takes an input the entire question and returns the answer after running pandas calculations""",
        ),
        Tool(
            name="Weather Agent",
            func=weather_agent_executor_wrapper,
            description="""useful when you need to answer questions about weather, current conditions, or forecasts.
                          Best for weather-related queries about specific locations.""",
        ),  
    ]

    grand_agent = create_react_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=grand_tools,
        prompt=base_prompt.partial(instructions="""You are a helpful assistant that can use multiple tools to help users.
        For weather-related queries, use the Weather Agent.
        For Python code execution and QR code generation, use the Python Agent.
        For analyzing episode data from CSV files, use the CSV Agent.
        Choose the most appropriate tool based on the user's request."""),
    )
    
    return AgentExecutor(
        agent=grand_agent,
        tools=grand_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )


def main():
    st.set_page_config(
        page_title="AI Agent Interface",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize the grand agent
    grand_agent_executor = setup_agents()

    # Sidebar
    with st.sidebar:
        st.title("🛠️ AI Agent Interface")
        task_type = st.radio(
            "Select Task Type",
            ["Episode Analysis", "QR Code Generation", "Weather Query", "Custom Query"],
            help="Choose the type of task you want to perform"
        )
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This interface provides access to:
        - Episode Analysis: Analyze CSV data
        - QR Code Generation: Create QR codes
        - Weather Query: Get weather information
        - Custom Query: Any other task
        """)

    # Main content area
    st.title("AI Task Assistant")
    st.markdown("---")

    if task_type == "Episode Analysis":
        st.header("📊 Episode Analysis")
        query = st.text_area(
            "Enter your question about the episodes:",
            placeholder="Example: Which season has the most episodes?",
            height=100
        )
        
        if st.button("Analyze", type="primary"):
            if query:
                with st.spinner("Analyzing episodes data..."):
                    result = grand_agent_executor.invoke({"input": query})
                    st.success("Analysis Result:")
                    st.write(result["output"])
            else:
                st.warning("Please enter a question about the episodes.")

    elif task_type == "QR Code Generation":
        st.header("🔲 QR Code Generator")
        num_codes = st.number_input(
            "Number of QR codes to generate",
            min_value=1,
            max_value=50,
            value=15
        )
        url = st.text_input(
            "URL for QR codes",
            value="www.udemy.com/course/langchain",
            placeholder="Enter the URL for the QR codes"
        )
        
        if st.button("Generate QR Codes", type="primary"):
            with st.spinner("Generating QR codes..."):
                result = grand_agent_executor.invoke({
                    "input": f"Generate and save in current working directory {num_codes} qrcodes that point to `{url}`"
                })
                st.success("QR Codes Generated:")
                st.write(result["output"])

    elif task_type == "Weather Query":
        st.header("🌤️ Weather Information")
        location = st.text_input(
            "Enter location:",
            placeholder="Example: New York, USA"
        )
        query_type = st.selectbox(
            "Select query type",
            ["Current Weather", "Weather Forecast", "Custom Weather Query"]
        )
        
        if st.button("Get Weather Info", type="primary"):
            if location:
                with st.spinner("Fetching weather information..."):
                    if query_type == "Current Weather":
                        query = f"What is the current weather in {location}?"
                    elif query_type == "Weather Forecast":
                        query = f"What is the weather forecast for {location}?"
                    else:
                        query = f"Tell me about the weather in {location}"
                        
                    result = grand_agent_executor.invoke({"input": query})
                    st.success("Weather Information:")
                    st.write(result["output"])
            else:
                st.warning("Please enter a location.")

    else:  # Custom Query
        st.header("🤖 Custom Query")
        st.info("You can ask any question or request any task. The AI will automatically choose the best agent to handle it.")
        
        query = st.text_area(
            "Enter your query:",
            placeholder="Enter any question or task...",
            height=100
        )
        
        if st.button("Execute", type="primary"):
            if query:
                with st.spinner("Processing your request..."):
                    result = grand_agent_executor.invoke({"input": query})
                    st.success("Result:")
                    st.write(result["output"])
            else:
                st.warning("Please enter a query first.")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by LangChain and OpenAI*")


if __name__ == "__main__":
    main()
