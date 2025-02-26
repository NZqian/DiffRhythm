import torch
import torch_npu
import torchaudio
from einops import rearrange
import argparse
import json
import os
from tqdm import tqdm
import random
import numpy as np

from diffrhythm.infer.utils import (
    get_reference_latent,
    get_lrc_token,
    extract_style_embeddings,
    prepare_model
)

def inference(cfm_model, vae_model, cond, text, duration, style_prompt, start_time):
    # import pdb; pdb.set_trace()
    with torch.inference_mode():
        generated, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            steps=32,
            cfg_strength=4.0,
            start_time=start_time
        )
        
        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2) # [b d t]
    
        output = vae_model.decode_audio(latent, chunked=True)

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        
        return output
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrc-path', type=str, default="example/eg.lrc") # lyrics of target song
    parser.add_argument('--ref-audio-path', type=str, default="example/eg.wav") # reference audio as style prompt for target song
    parser.add_argument('--audio-length', type=int, default=95) # length of target song
    parser.add_argument('--output-dir', type=str, default="example/output")
    args = parser.parse_args()
    
    device = 'npu'
    
    audio_length = args.audio_length
    if audio_length == 95:
        max_frames = 2048
    elif audio_length == 285:
        max_frames = 6144
    
    cfm, tokenizer, muq, vae = prepare_model()
    
    with open(args.lrc_path, 'r') as f:
        lrc = f.read()
    lrc_prompt, start_time = get_lrc_token(lrc, tokenizer, device)
    
    style_prompt = extract_style_embeddings(muq, args.ref_audio_path)
    
    latent_prompt = get_reference_latent(device, max_frames)
    
    generated_song = inference(cfm_model=cfm, 
                               vae_model=vae, 
                               cond=latent_prompt, 
                               text=lrc_prompt, 
                               duration=max_frames, 
                               style_prompt=style_prompt,
                               start_time=start_time
                               )
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "output.wav")
    torchaudio.save(output_path, generated_song, sample_rate=44100)
    