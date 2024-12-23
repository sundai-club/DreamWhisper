from fastapi import FastAPI, UploadFile, HTTPException
from openai import OpenAI
import os
from dotenv import load_dotenv
import shutil
from datetime import datetime

# Step 1: Initialize environment and configurations
load_dotenv(override=True) 

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key is missing. Set OPENAI_API_KEY in your environment variables.")
print(api_key)
# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Create necessary directories
TEMP_DIR = "temp_audio"
TRANSCRIPTS_DIR = "transcripts"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

app = FastAPI(
    title="DreamWhisper API",
    description="Dream recording and transcription API that processes audio files and saves transcripts for dream interpretation."
)

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile):
    try:
        print(f"Processing file: {audio.filename}")
        
        # Step 3: Save uploaded audio file temporarily
        temp_input_path = os.path.join(TEMP_DIR, f"input{os.path.splitext(audio.filename)[1]}")
        with open(temp_input_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        print(f"Saved file to {temp_input_path}, size: {os.path.getsize(temp_input_path)} bytes")
        
        # Step 4: Transcribe audio using OpenAI Whisper API
        try:
            with open(temp_input_path, "rb") as audio_file:
                print("Sending file to OpenAI API...")
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"  # Explicitly specify response format
                )
                print("Successfully received transcript from API")
        except Exception as api_error:
            print(f"API Error details: {str(api_error)}")
            raise

        # Step 5: Save transcript with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file_name = f"transcript_{timestamp}.txt"
        transcript_file_path = os.path.join(TRANSCRIPTS_DIR, transcript_file_name)

        # Save the transcript
        with open(transcript_file_path, "w") as f:
            f.write(transcript)

        # Step 6: Clean up temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

        # Step 7: Return transcript and file location
        return {"transcript": transcript, "file_path": transcript_file_path}

    except Exception as e:
        print(f"Error during transcription process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)