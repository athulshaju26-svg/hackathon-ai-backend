from fastapi import FastAPI, UploadFile, File
import whisper
import openai
import os
import json

app = FastAPI(title="Hackathon AI Summarizer")

# Load Whisper model
model = whisper.load_model("base")

# Use default API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    # Convert audio to text
    if file.content_type.startswith("audio"):
        path = file.filename
        with open(path, "wb") as f:
            f.write(await file.read())
        text = model.transcribe(path)["text"]
        os.remove(path)
    else:
        text = (await file.read()).decode()

    # Generate JSON summary
    prompt = f"""
    Summarize this conversation into JSON format:

    {text}

    Format:
    {{
      "summary": "",
      "topics": [],
      "action_items": [],
      "sentiment": ""
    }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)