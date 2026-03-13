import os
from google.adk.tools import ToolContext
import google.genai.types as types
from google.adk.agents.callback_context import CallbackContext
from os import path

from . import rag_pipeline

local_path = "application/temp"

async def save_file_as_artifact(content: str, tool_context: ToolContext):
    """Saves generated PDF report bytes as an artifact."""
    data_bytes = content.encode('utf-8')  # Convert string content to bytes
    report_artifact = types.Part(
        inline_data=types.Blob(mime_type="application/pdf", data=data_bytes),
    )
    filename = "save_artifact.pdf"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Save the bytes to a local file
    with open(path.join(local_path, filename), 'wb') as f:
        f.write(data_bytes)
    try:
        version = await tool_context.save_artifact(filename=filename, artifact=report_artifact)
        print(f"Successfully saved Python artifact '{filename}' as version {version}.")
        # The event generated after this callback will contain:
        # event.actions.artifact_delta == {"generated_report.pdf": version}
    except ValueError as e:
        print(f"Error saving Python artifact: {e}. Is ArtifactService configured in Runner?")
    except Exception as e:
        # Handle potential storage errors (e.g., GCS permissions)
        print(f"An unexpected error occurred during Python artifact save: {e}")

async def list_user_files_py(tool_context: ToolContext) -> str:
    """Tool to list available artifacts for the user."""
    try:
        available_files = await tool_context.list_artifacts()
        if not available_files:
            return "You have no saved artifacts."
        else:
            # Format the list for the user/LLM
            file_list_str = "\n".join([f"- {fname}" for fname in available_files])
            return f"Here are your available Python artifacts:\n{file_list_str}"
    except ValueError as e:
        print(f"Error listing Python artifacts: {e}. Is ArtifactService configured?")
        return "Error: Could not list Python artifacts."
    except Exception as e:
        print(f"An unexpected error occurred during Python artifact list: {e}")
        return "Error: An unexpected error occurred while listing Python artifacts."

def save_artifact_locally(filename: str, tool_context: ToolContext) -> dict:
    """
        Loads a saved artifact and saves it to a specified local path.
        ARGS:
        - filename: The name of the file to save.
        - tool_context: The tool context providing access to the artifact.

    """
    artifact = tool_context.load_artifact(filename)
    if artifact is None:
        return {"error": f"'{filename}' not found"}

    # Extract the data bytes
    data_bytes = artifact.inline_data.data
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Save the bytes to a local file
    with open(local_path, 'wb') as f:
        f.write(data_bytes)
        
    return {"status": "success", "message": f"Artifact saved to {local_path}"}

def save_doc_in_vecdb(content: str) -> None:
    """
    Save a document in the vector database.
    """
    rag = rag_pipeline.RAGPipeline()
    doc_id = rag.add_document(content)
    if doc_id==None:
        return {"status": "error", "message": "Failed to store document."}
    else:
        return {"status": "success", "message": f"Document stored with ID: {doc_id}"}

def retrieve_from_vecdb(query: str) -> None:
    """
    Retrieve context from the vector database.
    """
    rag = rag_pipeline.RAGPipeline()
    results = rag.retrieve(query)
    if not results['documents']:
        return {"status": "error", "message": "Document not found."}
    else:
        return {"status": "success", "message": f"Retrieved context: {results['documents']}"}