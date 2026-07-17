import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

load_dotenv()

google_api_key=os.getenv("gemini_api_key")

video_id="LPZh9BOjkQs"

try:
    ytt_api=YouTubeTranscriptApi()
    transcript_list=ytt_api.fetch(video_id,languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)
    #print(transcript)

except TranscriptsDisabled:
    print("No transcript available for this video")

splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
chunks=splitter.create_documents([transcript])

#print(len(chunks))

embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vector_store=FAISS.from_documents(chunks,embeddings)

#print(vector_store.index_to_docstore_id)

#retrieval

retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
print(retriever.invoke('What is LLM'))