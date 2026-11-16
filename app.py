import gradio as gr
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
        vibe = "Epic swashbuckling orchestral film score, driving low strings ostinato, bold French horns, thundering timpani drums, dramatic sea adventure tempo"
    elif any(word in poem_lower for word in ["love", "heart", "sad", "tears", "alone"]):
        vibe = "Melancholic cinematic cello solo, soft piano accompaniment, emotional nostalgia, slow romantic tempo"
    elif any(word in poem_lower for word in ["fight", "sword", "battle", "victory", "win"]):
        vibe = "Heroic epic orchestral crescendo, triumphant brass fanfare, aggressive percussion, fast-paced battle rhythm"
    else:
        vibe = "Dramatic cinematic ambient score, sweeping strings, gentle woodwinds, mysterious and atmospheric"
        
    return f"{vibe}, high quality studio recording, 120 BPM"

def generate_soundtrack(poem, duration=10):
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
    
    # Calculate max tokens based on duration (50 tokens per second is a rough estimate)
    max_new_tokens = int(duration * 50)
    
    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Step 3: Process output to WAV
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_data = audio_values[0, 0].numpy()
    
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save to file
    out_filename = "soundtrack.wav"
    scipy.io.wavfile.write(out_filename, rate=sampling_rate, data=audio_data)
    
    return out_filename, music_prompt

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
            generate_btn = gr.Button("Generate Cinematic Score", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Generated Soundtrack", type="filepath")
            prompt_output = gr.Textbox(label="AI Musical Prompt Generated", interactive=False)
            
    generate_btn.click(
        fn=generate_soundtrack, 
        inputs=[poem_input, duration_slider], 
        outputs=[audio_output, prompt_output]
    )

if __name__ == "__main__":
    demo.launch()