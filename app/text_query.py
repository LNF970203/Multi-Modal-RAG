import streamlit as st
import pinecone
import openai
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import os


MODEL = "text-embedding-ada-002"

# set up pinecone environment
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_TEXT_API_KEY"]
os.environ['PINECONE_API_ENV'] = "gcp-starter"
os.environ['PINECONE_INDEX_NAME'] = "multi-rag-text"
# set index
pinecone.init( api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_API_ENV'])
pinecone_index_text=pinecone.Index(os.environ['PINECONE_INDEX_NAME'])

# set the openai key
openai.api_key = st.secrets["OPENAI_API_KEY"]

def qa_engine(question):
    # pinecone env
    index=pinecone.Index(os.environ['PINECONE_INDEX_NAME'])

    question_embed_call = openai.embeddings.create(input = question ,model = MODEL)
    query_embeds = question_embed_call.data[0].embedding
    response = index.query(query_embeds,top_k = 1,include_metadata = True)
    # get the response text and metadata
    response = response["matches"][0]["metadata"]
    text = response.get("text", "UNKNOWN")
    chunk = response.get("chunk", "UNKNOWN")
    doc_index = response.get("doc_index", "UNKNOWN")
    offset=", OFFSET="+str(response.get("chunk","UNKNOWN"))

    # query document
    query_doc = []

    # create metadata for q&a chain
    metadata = {
        "id": chunk,
        "filename": doc_index,
        "source": str(doc_index) + offset
    }

    query_doc.append(Document(page_content=text, metadata = metadata))

    # query the answer from llm
    llm = OpenAI(openai_api_key = openai.api_key)
    chain = load_qa_with_sources_chain(llm, verbose = False)
    # get the chain response
    chain_response = chain.run(input_documents = query_doc, question = question )
    print(chain_response)
    return chain_response