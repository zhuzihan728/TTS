
from vits_ft_inference import inference
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import base64

app = FastAPI()

class UserRequest(BaseModel):
    text: str

def encode_base64_array(array: np.ndarray) -> str:
    # Encode the NumPy array to a base64 string
    return base64.b64encode(array.tobytes()).decode('utf-8')

@app.post("/vits_ft")
async def tts(request: UserRequest):
    text = request.text
    # Run inference to get the audio array and sampling rate
    audio, rate = inference(text)

    if audio is None or rate is None or not isinstance(audio, np.ndarray):
        raise HTTPException(status_code=500, detail="Error generating audio")

    # record dtype
    audio_dtype = str(audio.dtype)
    
    # convert to float64
    if audio_dtype != "float64":
        audio = audio.astype(np.float64)
    

    # Encode the NumPy array to a base64 string for transmission
    encoded_response_array = encode_base64_array(audio)

    # Prepare the response
    response = {
        "audio_dtype": audio_dtype,
        "sampling_rate": rate,
        "encoded_array": encoded_response_array
    }

    return response

if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=6969, app=app)