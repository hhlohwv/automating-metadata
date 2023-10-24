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
#from demo import read_single 

#TODO: IF doi -> then search open alex -> determine relevant metadata to return. -> Together once everything is up to date. 
#TODO: Combine Paper_data_Json_Single + Open Alex -> into a database_search -> to get external data. - Henry
#TODO: get api + langchain + sturcutred output in a pretty package -> Ellie
#TODO: Dockerize -> Ellie. 

#from ..Server.PDFDataExtractor.pdfdataextractor.demo import read_single
sys.path.append(os.path.abspath("/Users/desot1/Dev/automating-metadata/Server/PDFDataExtractor/pdfdataextractor"))
pyalex.config.email = "ellie@desci.com"

# Load in API keys from .env file
load_dotenv(find_dotenv())


def openalex(doi): 
    dict = Works()[doi]
    return dict

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

    # Elsevier API key
    apikey = os.getenv("apikey")
    client = httpx.Client()


    #%% Info from Crossref
    r = cr.works(ids = f'{doi}')  # Crossref search using DOI, "r" for request

    title = r['message']['title'][0]
    type = r['message']['type']
    pub_name = r['message']['container-title'][0]
    pub_date = r['message']['published']['date-parts'][0]
    #subject = r['message']['subject']

    inst_names = []  # handling multiple colleges, universities
    authors = []  # for handling multiple authors

    for i in r['message']['author']:
        authors.append(i['given'] + ' ' + i['family'])
        try:
            name = (i['affiliation'][0]['name'])
            if name not in inst_names:
                inst_names.append(name)
        except:
            continue

    if len(inst_names) == 0:  # returning message if no institutions returned by Crossref, may be able to get with LLM
        inst_names = "No institutions returned by CrossRef"


    refs = []
    for i in r['message']['reference']:
        try:
            refs.append(i['DOI'])
        except:
            refs.append(f"{i['key']}, DOI not present")
        
    url_link = r['message']['URL']
    

    #%% Info from Elsevier
    format = 'application/json'
    view ="FULL"
    url = f"https://api.elsevier.com/content/article/doi/{doi}?APIKey={apikey}&httpAccept={format}&view={view}"
    with httpx.Client() as client:
        r=client.get(url)
    
    json_string = r.text
    d = json.loads(json_string)  # "d" for dictionary

    try:
        d['full-text-retrieval-response']
        scopus_id = d['full-text-retrieval-response']['scopus-id']
        abstract = d['full-text-retrieval-response']['coredata']['dc:description']

        """keywords = []
        for i in d['full-text-retrieval-response']['coredata']['dcterms:subject']:
            keywords.append(i['$'])"""

        original_text = d['full-text-retrieval-response']['originalText']
    except:
        scopus_id = 'None, elsevier error'
        abstract = 'None, elsevier error'
        keywords = ['None, elsevier error']
        original_text = 'None, elsevier error'
    

    #%% Info from Semantic Scholar
    url = f'https://api.semanticscholar.org/graph/v1/paper/{doi}/?fields=fieldsOfStudy,tldr,openAccessPdf'
    with httpx.Client() as client:
        r = client.get(url)

    json_string = r.text
    d = json.loads(json_string)

    paper_id = d['paperId']

    field_of_study = []
    if d['fieldsOfStudy'] is None:
        field_of_study = 'None'
    else:
        for i in d['fieldsOfStudy']:
            field_of_study.append(i)
    if d['tldr'] is None:
        tldr = 'None'
    else:
        tldr = d['tldr']
    
    if d['openAccessPdf'] is None:
        openaccess_pdf = 'None'
    else:
        openaccess_pdf = d['openAccessPdf']['url']


    #%% Constructing output dictionary
    output_dict = {
        # Paper Metadata
        'title':title,
        'authors':authors,
        #'abstract':abstract,
        #'scopus_id':scopus_id,
        'paperId':paper_id,
        'publication_name':pub_name,
        'publish_date':pub_date,
        'type':type,
        #'keywords':keywords,
        #'subject':subject,
        'fields_of_study':field_of_study,
        'institution_names':inst_names,
        'references':refs,
        'tldr':tldr,
        #'original_text':original_text,
        'openAccessPdf':openaccess_pdf,
        'URL_link':url_link 
    }
   
    return output_dict


async def langchain_paper_search(file_path):
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
    loader = PyPDFLoader(str(file_path))  # convert path to string to work with loader
    document = loader.load_and_split()

    # Define all the queries and corresponding schemas in a list
    queries_schemas_docs = [
        ("What are the experimental methods and techniques used by the authors? This can include ways that data was collected as well as ways the samples were synthesized.", document),
        ("What is the scientific question, challenge, or motivation that the authors are trying to address?", document),
        ("Provide a summary of the results and discussions in the paper. What results were obtained and what conclusions were reached?", document),
        ("Provide a summary of each figure described in the paper. Your response should be a one sentence summary of the figure description, \
         beginning with 'Fig. #  - description...'. For example:'Fig. 1 - description..., Fig. 2 - description..., Fig. 3 - description...'. Separate each figure description by a single newline.", document),
        ("What future work or unanswered questions are mentioned by the authors?", document),
        ("Tell me who all the authors of this paper are. Your response should be a comma separated list of the authors of the paper, \
         looking like 'first author name, second author name", document)
    ]

    tasks = []

    # Run the queries concurrently using asyncio.gather
    for query, docs in queries_schemas_docs:
        task = chain.arun(doc_text=docs, query=query)
        # task = async_paper_search(query, docs, chain)
        tasks.append(task)

    summary = await asyncio.gather(*tasks)

    # Extracting individual elements from the summary
    methods, motive, results, figures, future, authors = summary  #NOTE: output to variables in strings

    llm_output = {
        "motive": motive,
        "method": methods,
        "figures": figures,
        "results": results,
        "future": future,
        "authors": authors
    }

    llm_output['figures'] = llm_output['figures'].split("\n")# using newline character as a split point.
    llm_output['authors'] = llm_output['authors'].split(', ') 

    llm_output['authors'] = get_orchid(llm_output["authors"])
    return llm_output

def get_orchid(authors): 
    orchid = []
    author_info = {}   
    print(type(authors))
    
    for author in authors: 
        #try: 
        url = "https://api.openalex.org/autocomplete/authors?q=" + author
        response = json.loads(requests.get(url).text)
        #except: 
        #    print("Your author might not be registered with ORCHID")
        
        if response["meta"]["count"] == 1: 
            orchid = response["results"][0]["external_id"]
            author_info[author] = {"orchid": orchid, "affiliation":response["results"][0]["hint"]}
        elif response["meta"]["count"] == 0: #FAKE - Create a test so we can check if the return is valid. 
            print("There are no ORCHID suggestions for this author")
        else: 
            orchid = response["results"][0]["external_id"]
            author_info[author] = {"orchid": orchid, "affiliation": response["results"][0]["hint"]}
            #create an async function which ranks the authors based on the similarity to the paper. 

    return author_info

def create_metadata_json(data):
    metadata_list = []

    def process_data(data):
        payload_body = {}
        for name, content in data.items():
            if isinstance(content, dict):
                payload_body[name] = process_data(content)
            else:
                payload_body[name] = content
        return payload_body

    for name, content in data.items():
        payload = {
            'name': name,
            'body': process_data(content) if isinstance(content, dict) else content
        }

        metadata = {
            'author': 'MetadataBot',
            'annotation': '',
            'payload': payload
        }

        metadata_list.append(metadata)

    return metadata_list


#%% Main, general case for testing
if __name__ == "__main__":
    #TODO: Update to be relevant. 

    print("Starting code run...")
    cwd = Path(__file__)
    pdf_folder = cwd.parents[1] #.joinpath('.test_pdf')  # path to the folder containing the pdf to test

    # File name of pdf in the .test_pdf folder for testing with code
    file_name = "Navahas2018.pdf"  # test 1
    # file_name = "Zhao et al_2023_Homonuclear dual-atom catalysts embedded on N-doped graphene for highly.pdf"  # test 2, too long
    # file_name = "Ren_Dong_2022_Direct electrohydrodynamic printing of aqueous silver nanowires ink on.pdf"  # test 3
    # file_name = "Chang et al_2022_Few-layer graphene as an additive in negative electrodes for lead-acid batteries.pdf"  # test 4
    # file_name = "Zhang et al_2019_Highly Stretchable Patternable Conductive Circuits and Wearable Strain Sensors.pdf"  # test 5
    # file_name = "Jepsen_2019_Phase Retrieval in Terahertz Time-Domain Measurements.pdf"  # test 6

    pdf_file_path = pdf_folder.joinpath(file_name)

    llm_output = asyncio.run(langchain_paper_search(pdf_file_path))  # output of unstructured text in dictionary
   
     
    json_result = create_metadata_json(llm_output)
    for item in json_result:
        print(json.dumps(item, indent=4))

    print(json_result)

    print("Script completed")
