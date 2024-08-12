"""
Incorporation of database searching and LLM text analysis for information extraction from scientific articles.
For LLM, using Ollama as the manager

Environment setup:
pip install langchain
pip install langchain-community
"""
from pathlib import Path

from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import ( # prompts for designing inputs
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings


#%% Define the Ollama model
model = "llama3"
llm = Ollama(model=model)


#%% Defining the chain using system and human prompts with variables
system_template = "You are a world class research assistant who produces answers based on facts. \
                        You are tasked with reading the following publication text and answering questions based on the following text: {context}.\
                        You do not make up information that cannot be found or interpreted from the text of the provided paper."

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)  # providing the system prompt

human_template = "{query}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = create_stuff_documents_chain(llm=llm, prompt=chat_prompt)


#%% Import PDF text and store in faiss index/database
file_path = Path("G:\My Drive\main\\02 Projects\LLM Document Analysis\\test_docs\Gao et al_2008_Total color difference for rapid and accurate identification of graphene.pdf")
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()
faiss_index = FAISS.from_documents(pages, OllamaEmbeddings(model=model))


#%% Use a question to perform a similarity/RAGS(?) search on the PDF text
# question = "What is the scientific motivation, challenge, or need that is being addressed?"
# question = "Is there a hypothesis or initial claim that is proposed, and if so what is it?"
# question = "What evidence is provided to support the hypotheses or claims presented?"
question = "What outside references are provided to support the hypotheses and claims presented? Please provide the associated citation information."
docs = faiss_index.similarity_search(question)

response = chain.invoke({'context':docs, 'query':question})

print(f"question: {question}")
print("")
print(f"Response: {response}")

# response = llm.invoke(f"What colors are mentioned in the following text: {text}")
# print(response)

# response = llm.invoke(f"What animals are mentioned in the following text: {text}")
# print(response)

# response = llm.invoke(f"What is the social dynamic between the sparrow, elephant, and hippo described in this text: {text}")
# print(response)