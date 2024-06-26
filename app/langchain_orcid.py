"""
Use of openai plus langchain for processing information in a pdf
Generated using chatGPT for incorporating asyncio for concurrent running of prompts
Generated by pasting my code from the analysis_v3 script with the following question:
Can you modify the below python code to incorporate asyncio to allow concurrent running of the paper_search() function?
"""
import sys 
import requests
import os
from pathlib import Path  # directory setting
import asyncio # For async asking of prompts
import json
from langchain.docstore.document import Document

import httpx  # for elsevier and semantic scholar api calls
from habanero import Crossref, cn  # Crossref database accessing
from dotenv import load_dotenv, find_dotenv  # loading in API keys
from langchain.document_loaders import PyPDFLoader # document loader import
from langchain.chat_models import ChatOpenAI  # LLM import
from langchain import LLMChain  # Agent import
from langchain.prompts.chat import ( # prompts for designing inputs
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.indexes import VectorstoreIndexCreator
import fitz #pdf reading library
import json
from pyalex import Works #, Authors, Sources, Institutions, Concepts, Publishers, Funders
import pyalex
import PyPDF2
import io
#import tiktoken
#from demo import read_single 

#TODO: IF doi -> then search open alex -> determine relevant metadata to return. -> Together once everything is up to date. 
#TODO: get api + langchain + sturcutred output in a pretty package -> Ellie

#from ..Server.PDFDataExtractor.pdfdataextractor.demo import read_single
sys.path.append(os.path.abspath("/Users/desot1/Dev/automating-metadata/Server/PDFDataExtractor/pdfdataextractor"))
pyalex.config.email = "ellie@desci.com"

# Load in API keys from .env file
load_dotenv(find_dotenv())


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_jsonld(node):
    base = "https://beta.dpid.org/"
    root = "?jsonld"

    #return requested node JSON
    response2 = requests.get(base+node+root).json()
    return response2

def get_pdf_text(node):
    base = "https://beta.dpid.org/"
    root = "?raw"

    #return most recent node JSON
    manifest = requests.get(base+node+root).json()
    
    #get the CID associated with the Payload + PDF object
    try:
        pdf_path = next(item['payload']['path'] for item in manifest['components'] if item['type'] == 'pdf')
        
        pdf_url = next(item['payload']['url'] for item in manifest['components'] if item['type'] == 'pdf')

    except: 
        return "No PDF object found"
    
    url = base+pdf_path+"?raw"
    
    ipfs="https://ipfs.desci.com/ipfs/"+pdf_url
  
    response = requests.get(ipfs) 
    
    if response.status_code == 200:
        # Open the PDF content with PyPDF2
        pdf_file = PyPDF2.PdfReader(io.BytesIO(response.content))
        
        # Check if the PDF is extractable
        if pdf_file.is_encrypted:
            pdf_file.decrypt("")  # Assuming the PDF has no password
            
        
        # Initialize text
        pdf_text = ""
        
        # Limit processing to under the relevant token limit
        num_pages = len(pdf_file.pages)
        pdf_word_count = 0

        for i in range(num_pages):
            page = pdf_file.pages[i]
            page_text = page.extract_text()
            
            if page_text is not None:
                page_word_count = len(page_text.split())
                total_word_count = pdf_word_count + page_word_count

                if total_word_count < 16000*.5:
                    pdf_text += page_text
                    pdf_word_count = total_word_count

                else: 
                    break
                 
        if pdf_word_count < 20: 
            print(f"Not enough information in the PDF from {url}. This could be because this PDF is rendered as an image, or doesn't have many words.")
            return None
            
        return pdf_text
    else:
        print(f"Error fetching PDF from {url}. Status code: {response.status_code}")
        return None

def paper_data_json_single(doi):
    """
    Create a json output file for a single paper using the inputed identifier.
    Only using a DOI string at the moment
    File constructed based on the info in metadata_formatting_categories.md

    Inputs:
    doi - string, DOI string for the paper/publication of interest
    output - string, path of where to save json output
    ---
    output:
    dictionary, conversion to json and writing to file
    """
    #%% Setting up info for usage of API's
    # define crossref object
    cr = Crossref()  
    cr.mailto = 'desotaelianna@gmail.com'
    cr.ua_string = 'Python/Flask script for use in Desci Nodes publication information retrieval.'


    #%% Info from Crossref
    try:
        r = cr.works(ids = f'{doi}')  # Crossref search using DOI, "r" for request
    except requests.exceptions.HTTPError as e:
        print(f"None, CrossRef DOI lookup returned error: {e}\n")

    try:
        title = r['message']['title'][0]
    except:
        title = f"None, Crossref Error"
    try:
        abstract = r['message']['abstract'][0]
    except:
        abstract = f"None, Crossref Error"
    try:
        type = r['message']['type']
    except:
        type = "None, Crossref Error"

    try:
        pub_name = r['message']['container-title'][0]
    except:
        pub_name = "None, Crossref Error"

    try:
        pub_date = r['message']['published']['date-parts'][0]
    except:
        pub_date = "None, Crossref Error"

    try:
        subject = r['message']['subject']
    except:
        subject = "None, Crossref Error"
    try:
        subject = r['message']['license']
    except:
        subject = "None, Crossref Error"
    
    authors_info = []

    for author in r['message']['author']:
        full_name = author['given'] + ' ' + author['family']
        authors_info.append(full_name)
    
    if authors_info:
        authors_info = get_orcid(authors_info)
    else: 
        authors_info.append = {'None, no authors returned by CrossRef'}


    refs = []
    for i in r['message']['reference']:
        try:
            refs.append(i['DOI'])
        except:
            refs.append(f"{i['key']}, DOI not present")
        
    url_link = r['message']['URL']
    
    

    #%% Info from Semantic Scholar
    url = f'https://api.semanticscholar.org/graph/v1/paper/{doi}/?fields=fieldsOfStudy,tldr,openAccessPdf'
    with httpx.Client() as client:
        r = client.get(url)

    json_string = r.text
    d = json.loads(json_string)

    try:
        paper_id = d['paperId']
    except:
        paper_id = "None, Semantic Scholar lookup error"

    field_of_study = []
    try:
        if d['fieldsOfStudy'] is None:
            field_of_study = 'None'
        else:
            for i in d['fieldsOfStudy']:
                field_of_study.append(i)
    except:
        field_of_study = "None, Semantic Scholar lookup error"

    try:
        if d['tldr'] is None:
            tldr = 'None'
        else:
            tldr = d['tldr']
    except:
        tldr = "None, Semantic Scholar lookup error"

    try:
        if d['openAccessPdf'] is None:
            openaccess_pdf = 'None'
        else:
            openaccess_pdf = d['openAccessPdf']['url']
    except:
        openaccess_pdf = "None, Semantic Scholar lookup error"

    # OpenAlex accessing as backup info for the previous tools
    try:
        openalex_results = Works()[doi]  # Crossref search using DOI, "r" for request
    except requests.exceptions.HTTPError as e:
        print(f"OpenAlex DOI lookup returned error: {e}\n")
        openalex = False
    
    if openalex: 

        if "error" in title:  # attempt replacing error title from cross with title from openalex
            try:
                title = openalex_results['title']
            except:
                pass
        if "error" in type:  # attempt replacing error keywords from cross with title from openalex
            try:
                type = openalex_results['type']
            except:
                pass
        if "error" in pub_name:  # attempt replacing error keywords from cross with title from openalex
            try:
                pub_name = openalex_results['primary_location']
            except:
                pass

        if "error" in pub_date:  # attempt replacing error keywords from cross with title from openalex
            try:
                pub_date = openalex_results['publication_date']
            except:
                pass
        
    try:
        openalex_id = openalex_results['id']
    except: 
        openalex_id = "None, OpenAlex Lookup error"
    try:
        keywords = openalex_results['keywords']
    except:
        keywords = "none"

    
    #%% Constructing output dictionary
    output_dict = {
        # Paper Metadata
        'title':title,
        'creator':authors_info,
        'datePublished':pub_date,
        'keywords':keywords,
        #'references':refs,
        #'tldr':tldr,
    }
    print("This is the output_dict \n", output_dict)
    return output_dict

async def langchain_paper_search(node):
    #file_path
    """
    Analyzes a pdf document defined by file_path and asks questions regarding the text
    using LLM's.
    The results are returned as unstructured text in a dictionary.
    """
    #%% Setup, defining framework of paper info extraction
    # Define language model to use
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

    # Defining system and human prompts with variables
    system_template = "You are a world class research assistant who produces answers based on facts. \
                        You are tasked with reading the following publication text and answering questions based on the information: {doc_text}.\
                        You do not make up information that cannot be found in the text of the provided paper."

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)  # providing the system prompt

    human_template = "{query}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=llm, prompt=chat_prompt)


    #%% Extracting info from paper
    # Define the PDF document, load it in
    text = get_pdf_text(node)
    document = Document(page_content = text)

    # Define all the queries and corresponding schemas in a list
    queries_schemas_docs = [
        #("What are the experimental methods and techniques used by the authors? This can include ways that data was collected as well as ways the samples were synthesized.", document),
        #("What is the scientific question, challenge, or motivation that the authors are trying to address?", document),
        #("Provide a summary of the results and discussions in the paper. What results were obtained and what conclusions were reached?", document),
        #("Provide a summary of each figure described in the paper. Your response should be a one sentence summary of the figure description, \
        # beginning with 'Fig. #  - description...'. For example:'Fig. 1 - description..., Fig. 2 - description..., Fig. 3 - description...'. Separate each figure description by a single newline.", document),
        #("What future work or unanswered questions are mentioned by the authors?", document),
        ("Tell me who all the authors of this paper are. Your response should be a comma separated list of the authors of the paper, \
         looking like 'first author name, second author name", document),
        ("Tell me the title of this paper", document)
    ]

    tasks = []

    # Run the queries concurrently using asyncio.gather
    for query, docs in queries_schemas_docs:
        task = chain.arun(doc_text=docs, query=query)
        tasks.append(task)

    summary = await asyncio.gather(*tasks)

    # Extracting individual elements from the summary
    #methods, motive, results, future figures, 
    authors, title = summary  #NOTE: output to variables in strings

    llm_output = {
        #"motive": motive,
        #"method": methods,
        #"figures": figures,
        #"results": results,
        #"future": future,
        "authors": authors,
        "title": title
    }

    #transform outputs
    #llm_output['figures'] = llm_output['figures'].split("\n")# using newline character as a split point.
    llm_output['authors'] = llm_output['authors'].split(', ') 
    llm_output['authors'] = get_orcid(llm_output["authors"])

    return llm_output
def get_orcid(authors):
    orcid_info = {}  # Dictionary to store author information
        
    for author in authors: 
        try: 
            url = "https://api.openalex.org/authors?search=" + author
            response = json.loads(requests.get(url).text)
        except Exception as e:  # Added variable 'e' to catch the exception
            print(f"OpenAlex ORCID lookup returned error: {e}\n")
            continue  # Skip to the next author
        
        if response["meta"]["count"] >= 1: 
            orcid = response["results"][0]["external_id"]
            affiliation = response["results"][0]["affiliations"][0]
            name = response["results"][0]["display_name"]

            author_info = {
                "orcid": orcid,
                "affiliation": affiliation
            }
        
            orcid_info[author] = author_info
            orcid_info[name] = orcid_info.pop(author)

        else:
            print("None, There are no OrcID suggestions for this author")
            author_info = "none"
            orcid_info[author] = author_info
            continue  # Skip to the next author

    return orcid_info

def update_json_ld(json_ld, new_data):
    # Process author information
    loop = 0
    for key, value in new_data.items(): 
        loop+=1
        if "None" in value:
            continue
        elif key == "authors": 
            authors = new_data.get("authors")

            for author_name, author_info in authors.items():

                creator_entry = {
                    "@type": "Person",
                    "name": author_name
                }
                if author_info is dict: 
                    orchid = author_info.get("orcid")
                    organization = author_info.get("affiliation")

                    if orchid:
                        creator_entry["@id"] = orchid
                    if organization:
                        creator_entry["organization"] = organization
                

                json_ld["@graph"].append(creator_entry)
                json_ld["@graph"][1]["creator"].append(creator_entry)
        else:
            json_ld["@graph"][1][key.lower()] = value
            print("I'm adding: " + str(value))

        #print(loop)
    return json_ld


#%% Main, general case for testing
def run(node, doi=None):

    print("Starting code run...")
    if node is not None:
        print(f"NODE is set to: {node}")
    else:
        print("NODE is not set.")

    json_ld = get_jsonld(node)

    if doi: 
        lookup_results = paper_data_json_single(doi)
        #updated_json_ld = update_json_ld(json_ld, lookup_results)
        
    else: 
        updated_json_ld = json_ld

    llm_output = asyncio.run(langchain_paper_search(node))# output of unstructured text in dictionary
    #updated_json_ld = update_json_ld(json_ld, llm_output)
    updated_json_ld = json_ld
   
    #doi = "https://doi.org/10.1002/adma.202208113"
    
    print("Script completed")

if __name__ == "__main__":
    run("46")