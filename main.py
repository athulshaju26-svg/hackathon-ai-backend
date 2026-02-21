from fastapi import FastAPI, UploadFile, File, Form
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/summarize")
async def summarize(
    file: UploadFile = File(None),
    text: str = Form(None)
):
    try:
        # STEP 1: Get text from audio if file uploaded
        if file:
            audio_bytes = await file.read()

            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=(file.filename, audio_bytes)
            )

            text = transcript.text

        # STEP 2: Summarize text
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize conversation into JSON with key points."},
                {"role": "user", "content": text}
            ]
        )

        return {"summary": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}
