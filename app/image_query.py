import streamlit as st
import os
import clip
import pinecone

DEVICE = "cpu"

# set up pinecone environment
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_IMAGE_API_KEY"]
os.environ['PINECONE_API_ENV'] = "gcp-starter"
os.environ['PINECONE_INDEX_NAME'] = "multi-rag-images"
# set index
pinecone.init( api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_API_ENV'])
pinecone_index_image=pinecone.Index(os.environ['PINECONE_INDEX_NAME'])


@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32")
    model = model.to(DEVICE)

    return model, preprocess


def query_images(query):
    model, _ = load_clip_model()
    # get the tekenizers
    tokens = clip.tokenize(query).to(DEVICE)
    query_embeds = model.encode_text(tokens).tolist()[0]
    response = pinecone_index_image.query(query_embeds,top_k = 1,include_metadata = True)
    file_path = response['matches'][0]['metadata']['file_path']
    image_name = response['matches'][0]['metadata']['image_name']
    score = response['matches'][0]['score']
    return file_path, score, image_name