# code-interpreter-llm

A multi-agent AI interface built with Streamlit, LangChain, and OpenAI that provides various functionalities including episode data analysis, QR code generation, and weather information queries.

## Features

- **Episode Analysis**: Analyze CSV data from episode_info.csv using natural language queries
- **QR Code Generation**: Create multiple QR codes with custom URLs
- **Weather Query**: Get current weather information and forecasts for any location
- **Custom Query**: Execute any custom task using the most appropriate AI agent

## Prerequisites

- Python 3.8+
- OpenAI API key
- Tavily API key (for weather searches)

## Dependencies

```bash
pip install streamlit
pip install python-dotenv
pip install langchain
pip install langchain-openai
pip install langchain-experimental
pip install langchain-community
pip install qrcode
pip install pandas
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/code-interpreter-llm.git
cd code-interpreter-llm
```

2. Create a `.env` file in the root directory and add your API keys: