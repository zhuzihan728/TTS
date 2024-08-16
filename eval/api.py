import numpy as np
import requests
import base64
import soundfile as sf

def encode_base64_array(array: np.ndarray) -> str:
    # Encode the NumPy array to a base64 string
    return base64.b64encode(array.tobytes()).decode('utf-8')

def decode_base64_array(encoded_array: str) -> np.ndarray:
    # Decode the base64 string to a NumPy array
    decoded_bytes = base64.b64decode(encoded_array)
    np_array = np.frombuffer(decoded_bytes, dtype=np.float64)
    return np_array

def call_vits_ft(text: str) :
    url = "http://127.0.0.1:6969/vits_ft"

    data = {
        "text": text
    }

    response = requests.post(url, json=data)
    response_data = response.json()
    # response = {
    #     "audio_dtype": audio_dtype,
    #     "sampling_rate": rate,
    #     "encoded_array": encoded_response_array
    # }
    decoded_response_array = decode_base64_array(response_data['encoded_array'])

    audio_dtype = response_data['audio_dtype']
    if audio_dtype != "float64":
        # convert float64 to str audio_dtype
        decoded_response_array = decoded_response_array.astype(np.dtype(audio_dtype))
        

    return decoded_response_array, response_data["sampling_rate"]

def to_wave(audio, rate, output_path):
    sf.write(output_path, audio, rate)

TTS_FNs = {call_vits_ft.__name__: call_vits_ft}

if __name__ == "__main__":
    text = "你好，我是琪亚娜！"
    audio, rate = call_vits_ft(text)
    to_wave(audio, rate, "output.wav")