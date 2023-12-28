import streamlit as st
from image_query import query_images
from text_query import qa_engine


# full pipeline
st.title("Multi-Modal-RAG-Chat-Interface")

# text input
query_text = st.text_input("Enter the Question")

if query_text:
    url, score, image_name = query_images(query_text)
    if url and score and image_name:
        st.toast("Image Query Successful 🖼️")
    answer = qa_engine(query_text)
    if answer:
        st.toast("Text Query Successful 📒")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Text Query")
        st.markdown("**{}**".format(query_text))
        st.markdown("**Answer**: {}".format(answer.split("SOURCES")[0]))
    with col2:
        st.subheader("Image Query")
        st.image(url, caption=f"{image_name}")
        st.write("**Score**: {}".format(score))