from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService 
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

from . import tools

# root_agent = Agent(
#     model="gemini-3-flash-preview",
#     name='root_agent',
#     description="Agent to analyse uploaded document, save and list.",
#     instruction='''
#     You are a document analyser agent. Your task is to analyze the content of uploaded documents, save them for future reference - 'use save_file_as_artifact tool'.
#     ''',
#     tools=[tools.save_file_as_artifact]
#)

root_agent = Agent(
    model="gemini-3-flash-preview",
    name='root_agent',
    description="Agent to save uploaded document",
    instruction='''
    You are a helpful assistant that answers user asked queries.
    If the user wants to question-answer on any content you should follow the below steps:
    1. Take content from the user. it can be either text or a file.
    2. Use the 'save_doc_in_vecdb' tool to save the content in the vectordb.
    3. before answering any user question regarding the content, Use the 'retrieve_from_vecdb' tool to get the relevant context.
    ''',
    tools=[tools.save_doc_in_vecdb, tools.retrieve_from_vecdb],
)

# Instantiate the desired artifact service
artifact_service = InMemoryArtifactService()

# Provide it to the Runner
runner = Runner(
    agent=root_agent,
    app_name="artifact_app",
    session_service=InMemorySessionService(),
    artifact_service=artifact_service # Service must be provided here
)