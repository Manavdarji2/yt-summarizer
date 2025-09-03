import streamlit as st
import langchain_youtube_video_loader as lyvl
import textwrap

st.title("Youtube Assitant")
with st.sidebar:
    with st.form(key="my_form"):
        youtube_url=st.sidebar.text_area(
            label="What is the YouTube video URL?",
            max_chars=60
        )
        query=st.sidebar.text_area(
            label="Ask me about the video",
            max_chars=60,
            key="query"
        )
        submit=st.form_submit_button(label="Submit")


if query and youtube_url:
    db=lyvl.create_vector_db_from_youtube_url(youtube_url)
    response=lyvl.get_response_from_query(db, query)
    st.subheader("Answer")
    st.markdown(response, unsafe_allow_html=True)