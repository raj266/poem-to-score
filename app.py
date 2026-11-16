try:
    import spaces
except ImportError:
    pass

import gradio as gr
import edge_tts
import asyncio
import subprocess
import torch
import numpy as np
import scipy.io.wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration, pipeline

# 1. Load MusicGen on startup
print("Loading MusicGen...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

def analyze_poem_and_orchestrate(poem):
    """A lightweight heuristic 'LLM' to extract musical prompts from poetry."""
    poem_lower = poem.lower()
    
    # Simple sentiment/theme extraction
    if any(word in poem_lower for word in ["storm", "sea", "ocean", "wave", "sail"]):
        vibe = "epic orchestral film score, dramatic sea adventure, heavy brass, thundering drums, 120 bpm"
    elif any(word in poem_lower for word in ["love", "heart", "sad", "tears", "alone"]):
        vibe = "sad cinematic cello, soft emotional piano, slow romantic, 70 bpm"
    elif any(word in poem_lower for word in ["fight", "sword", "battle", "victory", "win"]):
        vibe = "epic battle music, aggressive percussion, fast string ostinato, triumphant brass, 140 bpm"
    else:
        vibe = "cinematic ambient score, sweeping strings, mysterious, atmospheric, 90 bpm"
        
    return f"{vibe}, high quality studio recording, 120 BPM"

@spaces.GPU
def generate_soundtrack(poem, duration=10, add_voice=False):
    """Generates the audio track based on the poem."""
    if not poem.strip():
        return None, "Please enter a poem first."
        
    # Step 1: Get the prompt
    music_prompt = analyze_poem_and_orchestrate(poem)
    
    # Step 2: Generate Audio
    inputs = processor(
        text=[music_prompt],
        padding=True,
        return_tensors="pt",
    )
    
    max_new_tokens = int(duration * 50)
    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Step 3: Process output to WAV
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_data = audio_values[0, 0].numpy()
    audio_data = audio_data / np.max(np.abs(audio_data))
    audio_data = (audio_data * 32767).astype(np.int16)
    
    bg_filename = "background.wav"
    scipy.io.wavfile.write(bg_filename, rate=sampling_rate, data=audio_data)
    
    # Optional Step 4: Spoken Word TTS
    if add_voice:
        try:
            # Generate TTS using Edge-TTS
            voice_file = "voice.mp3"
            asyncio.run(edge_tts.Communicate(poem, "en-US-ChristopherNeural").save(voice_file))
            
            # Mix with FFmpeg (available on HF Spaces by default)
            out_filename = "cinematic_poem.mp3"
            # Mixing: lower background volume to 30%, keep voice at 100%
            subprocess.run([
                "ffmpeg", "-y", "-i", bg_filename, "-i", voice_file,
                "-filter_complex", "[0:a]volume=0.3[bg];[1:a]volume=1.0[voice];[bg][voice]amix=inputs=2:duration=first[a]",
                "-map", "[a]", out_filename
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_filename, music_prompt
        except Exception as e:
            print(f"TTS Error: {e}")
            # Fallback to just music if TTS fails
            return bg_filename, music_prompt + " (Voiceover failed, music only)"
            
    return bg_filename, music_prompt

# 3. Build the Gradio UI
with gr.Blocks(title="Poem to Cinematic Score") as demo:
    gr.Markdown("# 🎭 Poem to Cinematic Score")
    gr.Markdown("Paste a poem, and our AI will analyze its emotional arc to generate a custom orchestral soundtrack.")
    
    with gr.Row():
        with gr.Column():
            poem_input = gr.Textbox(
                lines=8, 
                placeholder="Roses are red, violets are blue...",
                label="Your Poem"
            )
            duration_slider = gr.Slider(minimum=5, maximum=15, value=10, step=1, label="Duration (Seconds)")
            voice_toggle = gr.Checkbox(label="Add Spoken-Word Narration", value=False)
            generate_btn = gr.Button("Generate Cinematic Score", variant="primary")
            
            # Add the One-Click Examples here
            gr.Examples(
                examples=[
                    ["The ocean raged against the black ship, a battle for the dawn. The storm will never break us.", 10, True],
                    ["Shall I compare thee to a summer's day? Thou art more lovely and more temperate.", 10, False],
                    ["Once upon a midnight dreary, while I pondered, weak and weary, over many a quaint and curious volume of forgotten lore.", 10, True]
                ],
                inputs=[poem_input, duration_slider, voice_toggle],
                label="Click to try a sample poem"
            )
            
        with gr.Column():
            audio_output = gr.Audio(label="Generated Soundtrack", type="filepath")
            prompt_output = gr.Textbox(label="AI Musical Prompt Generated", interactive=False)
            
    generate_btn.click(
        fn=generate_soundtrack, 
        inputs=[poem_input, duration_slider, voice_toggle], 
        outputs=[audio_output, prompt_output]
    )

def _noop():
    return None
if "spaces" in globals():
    _noop = spaces.GPU(_noop)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)