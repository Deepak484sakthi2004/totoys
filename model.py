from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.core import StorageContext
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from pgService import getContext

import sys

#==================================================================================================
# Initialize memory outside the function to maintain state across calls
memory = ConversationBufferMemory(input_key="input", memory_key="chat_history")


classFetch = "class8"

def findSubject(query):
    prompt_template = f"""
    system
    You are an intelligent Agent, your task is to find which subject does the question belongs to!.
    find the relevancy of the question eith these subjects and return a json with subject as a key
    the subjects are # english, maths, science, socialScience #
    Question: {query}
    Expected Answer: 
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    #model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyClo-Pfhrww33nYHWXNW_UbjXa7pVggGtM')
    model = init_llm()
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({"query": query}, return_only_outputs=True)
    return response["subject"]


def init_llm():
    llm = Ollama(model="llama3", request_timeout=300.0)
    return llm


def init_query_engine(user_question,expertLevel,context):
    llm = init_llm()
    prompt_template = f"""
        Chat History: {chat_history}
    system
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    Use three points minimum and keep the answer concise. <eot_id><start_header_id>user<end_head_id>
    Question: {user_question}
    Context: {context}
    Student_expertise_level : {expertLevel}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question", "chat_history"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyClo-Pfhrww33nYHWXNW_UbjXa7pVggGtM')
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    return chain

def chat_cmd(user_question,expertLevel):
    context = getContext(classFetch,user_question)
   # context = " ".join([doc.page_content for doc in docs])
    chain = init_query_engine(user_question,expertLevel,context)
    response = chain.predict(input=user_question, context=context)
    print(response)
    return response, context

    
