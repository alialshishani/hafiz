import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# 1. Initialize the Engine
processor = Wav2Vec2Processor.from_pretrained("elgeish/wav2vec2-large-xlsr-53-arabic")
model = Wav2Vec2ForCTC.from_pretrained("elgeish/wav2vec2-large-xlsr-53-arabic")

def audit_recitation(audio_path):
    # 2. Load the "Physical Data"
    speech, sr = torchaudio.load(audio_path)
    
    # 3. Resample if necessary (Standardize the Fuel)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech = resampler(speech)
    
    # 4. Perform the Inference (The Process)
    input_values = processor(speech.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
    
    with torch.no_grad():
        logits = model(input_values).logits

    # 5. Decode IDs to Arabic Text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

# RUN THE AUDIT
result = audit_recitation("recitation.wav")
print(f"\n--- AI Transcription ---\n{result}\n")
