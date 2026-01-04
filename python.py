import torch

def transcribe(audio_path):
    # 1. Load and Resample
    speech = process_recitation(audio_path)
    
    # 2. Convert to Tensors
    inputs = processor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    
    # 3. Perform Inference (The "Aha!" Moment)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    # 4. Decode the IDs to Arabic Text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

# To test it, you need a .wav file in the same folder
# print(transcribe("my_recitation.wav"))
