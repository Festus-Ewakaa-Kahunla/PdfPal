from pydantic import BaseModel
import pymongo
import traceback
import os
import sys
from fastapi import FastAPI, UploadFile, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import gc
import awswrangler as wr
import boto3
import uuid
from typing import List
from dotenv import load_dotenv

# Load environment variables from a `.env` file
load_dotenv(".env")

# Retrieve and assign environment variables to variables
S3_KEY = os.getenv("S3_KEY")  # AWS S3 access key
S3_SECRET = os.getenv("S3_SECRET")  # AWS S3 secret access key
S3_BUCKET = os.getenv("S3_BUCKET")  # AWS S3 bucket name
S3_REGION = os.getenv("S3_REGION")  # AWS S3 region
S3_PATH = os.getenv("S3_PATH")  # AWS S3 path
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API key
MONGO_URL = os.getenv("MONGO_URL")  # MongoDB connection URL

try:
    # Connect to the MongoDB using the provided MONGO_URL
    client = pymongo.MongoClient(MONGO_URL, uuidRepresentation="standard")
    # Access the "chat_with_doc" database
    db = client["chat_with_doc"]
    # Access the "chat-history" collection within the database
    conversationcol = db["chat-history"]

    # Create an index on the "session_id" field, ensuring uniqueness
    conversationcol.create_index([("session_id")], unique=True)
except:
    # Handle exceptions and print detailed error information
    print(traceback.format_exc())

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    # Print information about the exception type, filename, and line number
    print(exc_type, fname, exc_tb.tb_lineno)


class ChatMessageSent(BaseModel):
    session_id: str = None
    user_input: str
    data_source: str

def get_response(
    file_name: str,
    session_id: str,
    query: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
):
    print("file name is ", file_name)
    file_name = file_name.split("/")[-1]
    
    embeddings = OpenAIEmbeddings()  # load embeddings
    
    # Download file from S3
    wr.s3.download(path=f"s3://{S3_BUCKET}/{S3_PATH}{file_name}", local_file=file_name, boto3_session=aws_s3)

    if file_name.endswith(".docx"):
        loader = Docx2txtLoader(file_path=file_name.split("/")[-1])
    else:
        loader = PyPDFLoader(file_name)

    # Load data
    data = loader.load()
    # Split data so it can fit GPT token limit
    print("splitting ..")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=["\n", " ", ""]
    )

    all_splits = text_splitter.split_documents(data)
    # Store data in vector db to conduct search
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    # Init OpenAI
    llm = ChatOpenAI(model_name=model, temperature=temperature)

    # Pass the data to OpenAI chain using vector db
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
    )
    
    with get_openai_callback() as cb:
        answer = qa_chain(
            {
                "question": query,  # User query
                "chat_history": load_memory_to_pass(
                    session_id=session_id
                ),  # Pass chat history for context
            }
        )
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        answer["total_tokens_used"] = cb.total_tokens
    gc.collect()  # Collect garbage from memory
    return answer

def load_memory_to_pass(session_id: str):
    """
    Load conversation history for a given session ID.
    """
    data = conversationcol.find_one(
        {"session_id": session_id}
    )  # Find the document with the session id
    history = []  # Create empty array (in case we do not have any history)
    if data:  # Check if data is not None
        data = data["conversation"]  # Get the conversation field

        for x in range(0, len(data), 2):  # Iterate over the field
            history.extend(
                [(data[x], data[x + 1])]
            )  # Our history is expected format is [(human_message, ai_message)], the even index has human message and odd has AI response
    print(history)
    return history  # Return history

def get_session() -> str:
    """
    Generate a new session ID.
    """
    return str(uuid.uuid4())

def add_session_history(session_id: str, new_values: List):
    """
    Add conversation history to an existing session or create a new session.
    """
    document = conversationcol.find_one(
        {"session_id": session_id}
    )  # Find the document with the session id
    if document:  # Check if data is not None
        # Extract the conversation list
        conversation = document["conversation"]

        # Append new values
        conversation.extend(new_values)

        # Update the document with the modified conversation list (for old session), we use update_one
        conversationcol.update_one(
            {"session_id": session_id}, {"$set": {"conversation": conversation}}
        )
    else:
        conversationcol.insert_one(
            {
                "session_id": session_id,
                "conversation": new_values,
            }  # To initiate a history under a new session, note we use insert_one
        )


# Create a FastAPI application
app = FastAPI()

# Add CORS middleware to handle Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=False,  # Allow sending credentials (e.g., cookies)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Create an AWS S3 session with provided access credentials
aws_s3 = boto3.Session(
    aws_access_key_id=S3_KEY,  # Set the AWS access key ID
    aws_secret_access_key=S3_SECRET,  # Set the AWS secret access key
    region_name=S3_REGION,  # Set the AWS region
)

@app.post("/chat")
async def create_chat_message(
    chats: ChatMessageSent,
):
    """
    Create a chat message and obtain a response based on user input and session.
    """
    try:
        if chats.session_id is None:
            session_id = get_session()

            payload = ChatMessageSent(
                session_id=session_id,
                user_input=chats.user_input,
                data_source=chats.data_source,
            )
            payload = payload.model_dump()

            response = get_response(
                file_name=payload.get("data_source"),
                session_id=payload.get("session_id"),
                query=payload.get("user_input"),
            )

            add_session_history(
                session_id=session_id,
                new_values=[payload.get("user_input"), response["answer"]],
            )

            return JSONResponse(
                content={
                    "response": response,
                    "session_id": str(session_id),
                }
            )

        else:
            payload = ChatMessageSent(
                session_id=str(chats.session_id),
                user_input=chats.user_input,
                data_source=chats.data_source,
            )
            payload = payload.dict()

            response = get_response(
                file_name=payload.get("data_source"),
                session_id=payload.get("session_id"),
                query=payload.get("user_input"),
            )

            add_session_history(
                session_id=str(chats.session_id),
                new_values=[payload.get("user_input"), response["answer"]],
            )

            return JSONResponse(
                content={
                    "response": response,
                    "session_id": str(chats.session_id),
                }
            )
    except Exception:
        print(traceback.format_exc())

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail="error")


@app.post("/uploadFile")
async def uploadtos3(data_file: UploadFile):
    """
    Uploads a file to Amazon S3 storage.
    """
    print(data_file.filename.split("/")[-1])
    try:
        with open(f"{data_file.filename}", "wb") as out_file:
            content = await data_file.read()  # async read
            out_file.write(content)  # async write
        wr.s3.upload(
            local_file=data_file.filename,
            path=f"s3://{S3_BUCKET}/{S3_PATH}{data_file.filename.split('/')[-1]}",
            boto3_session=aws_s3,
        )
        os.remove(data_file.filename)
        response = {
            "filename": data_file.filename.split("/")[-1],
            "file_path": f"s3://{S3_BUCKET}/{S3_PATH}{data_file.filename.split('/')[-1]}",
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Item not found")

    return JSONResponse(content=response)


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app)
