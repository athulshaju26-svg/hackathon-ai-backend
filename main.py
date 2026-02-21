from fastapi import FastAPI, UploadFile, File, Form
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/summarize")
async def summarize(
    input_text: str = Form(None),
    input_file: UploadFile = File(None)
):
    try:
        # STEP 1: Determine input type
        if input_file:
            audio_bytes = await input_file.read()

            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=(input_file.filename, audio_bytes)
            )

            text = transcript.text

        elif input_text:
            text = input_text

        else:
            return {"error": "Please provide either text or an audio file."}

        # STEP 2: Summarize
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize conversation into clear JSON key points."},
                {"role": "user", "content": text}
            ]
        )

        return {"summary": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}
