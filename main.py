import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI

load_dotenv()

google_api_key=os.getenv("gemini_api_key")

video_id="LPZh9BOjkQs"

try:
    ytt_api=YouTubeTranscriptApi()
    transcript=ytt_api.fetch(video_id,languages=["en"])
    transcripts = " ".join(chunk.text for chunk in transcript)
    print(transcript)

except TranscriptsDisabled:
    print("No transcript available for this video")