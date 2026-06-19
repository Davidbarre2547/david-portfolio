import os
import re
import base64
import datetime

import pandas as pd
import streamlit as st


from io import BytesIO
from google import genai
from pytube import YouTube
from langchain_community.vectorstores import FAISS
# from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


from dotenv import load_dotenv


import nest_asyncio
nest_asyncio.apply()


# Loading environment Variables
load_dotenv()


# Configure GEMINI-API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please add it your .env")
    st.stop()


client = genai.Client()


# Set up Page Configuration
st.set_page_config(page_title="YouTube Conversationalist", page_icon="📼", layout="wide")


# Function to get download link from conversations
def get_download_link(content, filename, text):
    b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    href = (
        f'<a href="data:text/plain;base64,{b64}" '
        f'download="{filename}">{text}</a>'
    )
    return href


def get_download_link_excel(df, filename, text):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    mime = (
        "data:application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet;base64"
    )
    href = f'<a href="{mime},{b64}" download="{filename}">{text}</a>'
    return href


# Function to extract video ID from URL
def extract_video_id(video_url:str)->str:
    # Handle different YouTube URL formats
    patterns = [
        r'(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})'
    ]


    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)

    # If it's already just a video ID
    if len(video_url) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', video_url):
        return video_url

    return ""


# Function to get video transcript
def get_video_transcript(video_url, languages=None, preserve_formatting=False):
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Could not extract video ID from URL")
            return None

        # Using You_Transcript_API
        api = YouTubeTranscriptApi()


        if languages:
            # Fetch transcript in preferred languages list
            fetched = api.fetch(video_id, languages=languages,
            preserve_formatting=preserve_formatting)
        else:
            fetched = api.fetch(video_id, preserve_formatting=preserve_formatting)


        # Fetched is iterable list of snippet objects
        raw_data = fetched.to_raw_data()


        # Get video info with pytube
        try:
            yt = YouTube(video_url)
            st.session_state.video_title = yt.title
            st.session_state.video_thumbnail = yt.thumbnail_url
        except Exception:
            # If pytube fails, use default values
            st.session_state.video_title = f"Video {video_id}"
            st.session_state.video_thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


        st.session_state.video_url = video_url


        # Create segments from transcript
        segments = [ ]
        for i, item in enumerate(raw_data):
            segments.append({
                "index": i+1,
                "start_seconds": item["start"],
                "text": item["text"]
            })


        # Save segments to session state
        st.session_state.transcript_segments = segments


        # Create full transcript
        full_transcript = " ".join([item["text"] for item in raw_data])


        return full_transcript
    except Exception as e:
        st.error(f"Error retrieving transcript: {str(e)}")
        return None



# Function to create vector store from transcript
def create_vector_store(transcript):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )


    chunks = text_splitter.split_text(transcript)


    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)


    return vector_store


# Function to get relevant context from vector store
def get_context(query, vector_store, k=3):
    if not vector_store:
        return "No transcript data available."
    try:
        docs = vector_store.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        st.error(f"Error getting context: {str(e)}")
        return "Error retrieving context"

# Function to find relevant transcript segments for Audio Playback
def find_relevant_segments(query, vector_store, transcript_segments, k=3):
    if not vector_store or not transcript_segments:
        return None

    try:
        docs_with_scores = vector_store.similarity_search_with_score(query, k=k)


        if not docs_with_scores:
            return None

        best_doc = docs_with_scores[0][0].page_content


        # Find segment containing this text
        for segment in transcript_segments:
            if best_doc in segment["text"]:
                return segment

        # If no exact search is found, try find a segment with significant overlay
        for segment in transcript_segments:
            words_in_segment = set(segment["text"].lower().split())
            words_in_doc = set(best_doc.lower().split())
            common_words = words_in_segment.intersection(words_in_doc)


            if len(common_words) > 5: # Arbitary threshold
                return segment

        return None
    except Exception as e:
        st.error(f"Error finding relevant segments: {str(e)}")
        return None




# Function to generate response from Gemini
def generate_response(query, context):
    try:
        prompt=f"""
You're an AI assisstant that helps users understand YouTube video content.
Answer the question based on the following context from the video transcript.


Context from video:
{context}


User Question: {query}


Provide a helpful, accurate, and concise response. It the context does'nt contain
relevant information to answer the question, aacknowledge that and provide general
information that might be helpful.
"""
        response = client.models.generate_content(model="models/gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Error generating response"

#-------------------SESSION-STATE--------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "transcript_segments" not in st.session_state:
    st.session_state.transcript_segments = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "video_title" not in st.session_state:
    st.session_state.video_title = None
if "video_thumbnail" not in st.session_state:
    st.session_state.video_thumbnail = None
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
#-------------------SESSION-STATE--------------------
# Main Streamlit code
st.title("YouTube Conversationalist")
st.caption(f"Logged in at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Sidebar for video ID output
with st.sidebar:
    st.header("Video Input")
    st.markdown("""
Enter the YouTube video link or ID.
For example: `https://www.youtube.com/watch?v=flnVc2Ke-bw` or
`https://youtu.be/flnVc2Ke-bw`
""")
    video_input = st.text_input("Enter YouTube video link or ID",
    placeholder="https://www.youtube.com/watch?v=flnVc2Ke-bw",
    key="video_input", help="Enter the YouTube video link or ID")


    if st.button("Process Video"):
        with st.spinner("Processing video transcript and create embeddings"):
            try:
                if not video_input:
                    st.error("Please enter a YouTube video link or ID")
                elif "youtube.com" not in video_input and "youtu.be" not in video_input and len(video_input) != 11:
                    st.error("Pleae enter a valid YouTube URL or 11 character video id")
                else:
                    st.session_state.video_id = extract_video_id(video_input)
                    transcript = get_video_transcript(video_input)


                    if transcript:
                        st.session_state.vector_store = create_vector_store(transcript)
                        st.session_state.video_processed = True
                        # Clear previous conversations
                        st.session_state.messages = []
                        st.success("Video processed successfully!")
                    else:
                        st.error("Failed to retrieve transcript. Please try again.")
            except Exception as e:
                st.error(f"Error fetching transcript: {str(e)}")


    if st.session_state.video_processed and st.session_state.video_url:
        st.divider()
        st.subheader("Current Video")
        st.image(st.session_state.video_thumbnail, caption=st.session_state.video_title, width=250)
        st.write(st.session_state.video_title)


        # Download conversation option
        if st.session_state.messages:
            st.subheader("Download Options")
            # Prepare conversation text
            conversation_text = f"Conversation about: {st.session_state.video_title}\n"
            conversation_text += f"Video URL: {st.session_state.video_url}\n"
            conversation_text += f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n\n"


            for msg in st.session_state.messages:
                role = "You" if msg["role"] == "user" else "Assistant"
                conversation_text += f"{role}: {msg['content']}\n\n"


            # Download it as text
            filename = f"conversation_{st.session_state.video_id}_{pd.Timestamp.now().strftime('%Y-%m-%d_%I:%M:%S_%p')}.txt"


            st.markdown(get_download_link(conversation_text,
            filename, "Download Conversation as Text"), unsafe_allow_html=True)


            # Download as Excel
            try:
                conversation_data = []
                for msg in st.session_state.messages:
                    conversation_data.append({
                        "Role": "User" if msg["role"] == "user" else "Assistant",
                        "Content": msg["content"],
                        "Timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %I:%M:%S %p')
                    })


                conversation_df = pd.DataFrame(conversation_data)
                excel_filename = f"conversation_{st.session_state.video_id}_{pd.Timestamp.now().strftime('%Y-%m-%d_%I:%M:%S_%p')}.xlsx"


                st.markdown(get_download_link_excel(conversation_df, excel_filename, "Download Conversation as Excel"), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error creating Excel file: {str(e)}")


# Main chat Interface
if st.session_state.video_processed:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


            # If it is an assistant message, check if there's relevent segment for playback
            if message.get("role") == "assistant" and "question_for_playback" in message:
                question = message["question_for_playback"]
                segment = find_relevant_segments(
                question,
                st.session_state.vector_store,
                st.session_state.transcript_segments
                )


                if segment:
                    start_time = int(segment["start_seconds"])
                    video_id_for_embed = extract_video_id(st.session_state.video_url) or st.session_state.video_id
                    embed_url = f"https://www.youtube.com/embed/{video_id_for_embed}?start={start_time}"


                    with st.expander("Watch the relevant video segment"):
                        st.components.v1.iframe(embed_url, width=540, height=360)



    # User Query
    user_query = st.chat_input("Ask something about the video...")


    if user_query:
        try:
            # Add user message to chat history
            st.session_state.messages.append({"role":"user", "content":user_query})


            # Display user messae in chat
            with st.chat_message("user"):
                st.markdown(user_query)


            # Generate Response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing transcript and generating response..."):
                    context = get_context(
                        user_query,
                        st.session_state.vector_store
                    )


                    # Generate Response
                    response = generate_response(user_query, context)


                    # Display Response
                    st.markdown(response)


                    # Try to find a relevant segment for playback
                    segment = find_relevant_segments(
                        user_query,
                        st.session_state.vector_store,
                        st.session_state.transcript_segments
                    )


                    if segment:
                        start_time = int(segment["start_seconds"])
                        video_id_for_embed = extract_video_id(
                            st.session_state.video_url
                        ) or st.session_state.video_id
                        embed_url = f"https://www.youtube.com/embed/{video_id_for_embed}?start={start_time}"


                        with st.expander("Watch relevant video segment"):
                            st.components.v1.iframe(embed_url, width=540, height=360)


            # Add assistant response to chat history with the questions for playback reference
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "question_for_playback": user_query
            })


        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I'm sorry, I encountered an error while processing your question. Please try again later."
            })
else:
    st.info("Enter a YouTube video url or id in the sidebar to get started.")


# Footer
st.divider()
st.caption("Built with ❣️ with Streamlit, Gemini API, Langchain, 🤗Face and FAISS")