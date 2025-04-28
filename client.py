import os
import asyncio
from langchain_openai import AzureChatOpenAI
from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_resources
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="AI Assistant", page_icon="🤖")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content=(
            "You are a helpful assistant that works with Graphviz diagrams using predefined tools. After every change to the graph, render it out."
        ))
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LLM
lc_llm = AzureChatOpenAI(
    model_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    openai_api_key=os.environ["AZURE_OPENAI_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_VERSION"],
)


# WATSONX_APIKEY = os.getenv('WATSONX_APIKEY', "")
# WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID', "")

# lc_llm = ChatWatsonx(
#     model_id="mistralai/mistral-large",# "ibm/granite-3-8b-instruct",
#     url = "https://us-south.ml.cloud.ibm.com",
#     apikey = WATSONX_APIKEY,
#     project_id = WATSONX_PROJECT_ID,
#     params = {
#         "decoding_method": "greedy",
#         "temperature": 0,
#         "min_new_tokens": 5,
#         "max_new_tokens": 100000
#     }
# )

# Async query processor
async def process_query(query):
    async with MultiServerMCPClient(
        {
            "Graphviz": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
            
        }
    ) as client:
        tools = client.get_tools()
        agent = create_react_agent(lc_llm, tools)

        st.session_state.chat_history.append(HumanMessage(content=query))
        response = await agent.ainvoke({"messages": st.session_state.chat_history})
        ai_message = response["messages"][-1]
        # print(ai_message)
        st.session_state.chat_history.append(ai_message)

        return ai_message.content

# Sync wrapper
def run_async_task(query):
    return asyncio.run(process_query(query))

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input bar
user_input = st.chat_input("What can I graph for you?")

if user_input:
    # Display user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process input and get response
    response = run_async_task(user_input)

    # Display assistant's message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)


# Display image in the sidebar if it exists
if os.path.exists("/home/mihirkestur/2025-April-IBM-Lean-AI-Solutions-Hackathon/server/test.gv.png"):
    st.sidebar.image("/home/mihirkestur/2025-April-IBM-Lean-AI-Solutions-Hackathon/server/test.gv.png", caption="Graphviz Output", use_container_width=True)

# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
# from dotenv import load_dotenv
# import os
# import asyncio

# load_dotenv()

# # Initialize LLM once
# lc_llm = AzureChatOpenAI(
#     model_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
#     openai_api_key=os.environ["AZURE_OPENAI_KEY"],
#     openai_api_version=os.environ["AZURE_OPENAI_VERSION"],
# )

# # Initialize conversation history
# chat_history = [
#     SystemMessage(content=(
#         "You have access to multiple tools that can help answer queries. "
#         "Use them dynamically and efficiently based on the user's request."
#     ))
# ]

# async def main():
#     async with MultiServerMCPClient(
#         {
#             # "math": {
#             #     "command": "python",
#             #     "args": ["/home/mihirkestur/2025-April-IBM-Lean-AI-Solutions-Hackathon/server/graphviz_server.py"],
#             #     "transport": "stdio",
#             # },
#             "math": {
#                 "url": "http://localhost:8000/sse",
#                 "transport": "sse",
#             },
#         }
#     ) as client:
#         tools = client.get_tools()
#         agent = create_react_agent(lc_llm, tools)

#         while True:
#             query = input("Query: ")
#             if query.lower() in ("exit", "quit"):
#                 break

#             chat_history.append(HumanMessage(content=query))
#             response = await agent.ainvoke({"messages": chat_history})

#             ai_message = response["messages"][-1]
#             chat_history.append(ai_message)

#             print(f"Assistant: {ai_message}")

# # Run async loop
# if __name__ == "__main__":
#     asyncio.run(main())