from f5_tts.model import DiT, CFM
from f5_tts.infer.utils_infer import (
    load_model,
    load_checkpoint
)
import torch
import torch_npu
import torchaudio
from einops import rearrange
import tiktoken
# from prefigure.prefigure import get_all_args
import argparse
import json
import os
from tqdm import tqdm
import random
import numpy as np

import time as sys_time

from transformers import T5EncoderModel, AutoTokenizer

from f5_tts.model.modules import (
    precompute_freqs_cis,
    get_pos_embed_indices,
)

filter_keyword_list = [
    "纯音乐",
    "编曲",
    "作词",
    "作曲",
    "调音",
    "制作人",
    "录音师",
]

filter_full_list = [
    "music",
    "end"
]

def check_lyric(time: float, lyric: str):
    if time < 0.1:
        return False
    for filter_keyword in filter_keyword_list:
        if filter_keyword in lyric:
            return False
    for filter_full in filter_full_list:
        if filter_full == lyric.strip().lower():
            return False
    if len(lyric) == 0:
        return False
    return True

def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.strip()
    for line in lyrics.split('\n'):
        try:
            time, lyric = line[1:9], line[10:]
            lyric = lyric.strip()
            mins, secs = time.split(':')
            secs = int(mins) * 60 + float(secs)
            lyrics_with_time.append((secs, lyric))
        except:
            #traceback.print_exc()
            continue
            #print("error", line)
    return lyrics_with_time

class CNENTokenizer():
    def __init__(self):
        with open('./f5_tts/g2p/g2p/vocab.json', 'r') as file:
            self.phone2id:dict = json.load(file)['vocab']
        self.id2phone = {v:k for (k, v) in self.phone2id.items()}
        from f5_tts.g2p.g2p_generation import chn_eng_g2p
        self.tokenizer = chn_eng_g2p
    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x+1 for x in token]
        return token
    def decode(self, token):
        return "|".join([self.id2phone[x-1] for x in token])

def extract_t5_embeddings(model, tokenizer, text, max_length=128, device="cpu", project_out=False):
    model_dims = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
    }

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)  

    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        embeddings = model(input_ids=input_ids)["last_hidden_state"]

    if project_out:
        proj_out = torch.nn.Linear(model_dims[model.config.name_or_path], model_dims[model.config.name_or_path]) 
        embeddings = proj_out(embeddings.float())

    return embeddings

def inference(model, cond, text, duration, style_prompt, style, output_dir, song_name, ckpt_step, start_time, latent_pred_start_frame, latent_pred_end_frame, epoch, cfg_strength):
    # import pdb; pdb.set_trace()
    with torch.inference_mode():
        s_t = sys_time.time()
        generated, _ = model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            steps=32,
            cfg_strength=cfg_strength,
            sway_sampling_coef=None,
            start_time=start_time,
            latent_pred_start_frame=latent_pred_start_frame,
            latent_pred_end_frame=latent_pred_end_frame
        )
        e_t = sys_time.time() - s_t
        print(f"infer cost {e_t} s")
        # print(type(generated))
        # print(generated.shape)
        generated = generated.to(torch.float32)
        #torch.save(generated, "test.pt")
        latent = generated
        latent = latent.transpose(1, 2)
        
        # basename = f"{song_name}_{'_'.join(style.split(','))}_{ckpt_step}.pt"
        basename = f"{song_name}_{style}_{ckpt_step}_{epoch}.pt"
        output_path = os.path.join(output_dir, basename)
        
        torch.save(latent, output_path)
        
        return


def get_style_prompt(device, song_name, ref_npy_pth):
    mulan_style_path = ref_npy_pth
    mulan_stlye = np.load(mulan_style_path)
    
    style_prompt = torch.from_numpy(mulan_stlye).to(device) # [1, 512]
    style_prompt = style_prompt.half()
    
    return style_prompt
    


def get_lrc_prompt(text, tokenizer, dit_model, max_secs):

    max_frames = 2048
    lyrics_shift = 2
    sampling_rate = 44100
    downsample_rate = 2048
   
    pad_token_id = 0
    comma_token_id = 1
    period_token_id = 2    
    
    fsmin = -10
    fsmax = 10

    lrc_with_time = parse_lyrics(text)
    
    modified_lrc_with_time = []
    for i in range(len(lrc_with_time)):
        time, line = lrc_with_time[i]
        # line_token = self.tokenizer.encode(line)
        line_token = tokenizer.encode(line)
        modified_lrc_with_time.append((time, line_token))

    lrc_with_time = modified_lrc_with_time

    lrc_with_time = [(time_start, line) for (time_start, line) in lrc_with_time if time_start < max_secs]
    # latent_end_time = lrc_with_time[-1][0] if len(lrc_with_time) >= 1 else -1
    lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time
    
    normalized_start_time = 0.

    lrc = torch.zeros((max_frames,), dtype=torch.long)

    tokens_count = 0
    last_end_pos = 0
    for time_start, line in lrc_with_time:
        tokens = [token if token != period_token_id else comma_token_id for token in line] + [period_token_id]
        tokens = torch.tensor(tokens, dtype=torch.long)
        num_tokens = tokens.shape[0]

        gt_frame_start = int(time_start * sampling_rate / downsample_rate)
        
        frame_shift = random.randint(int(fsmin), int(fsmax))
        
        frame_start = max(gt_frame_start - frame_shift, last_end_pos)
        frame_len = min(num_tokens, max_frames - frame_start)

        #print(gt_frame_start, frame_shift, frame_start, frame_len, tokens_count, last_end_pos, full_pos_emb.shape)

        lrc[frame_start:frame_start + frame_len] = tokens[:frame_len]

        tokens_count += num_tokens
        last_end_pos = frame_start + frame_len   
        
    lrc_emb = lrc.unsqueeze(0).to(dit_model.device)
    
    normalized_start_time = torch.tensor(normalized_start_time).unsqueeze(0).to(dit_model.device)
    
    return lrc_emb, normalized_start_time

def get_latent_prompt(model, use_ref_latent, downsample_rate, song_name, max_frames, song_name2ref_latent):
    sampling_rate = 44100
    if use_ref_latent:
        pth_st_et = song_name2ref_latent[song_name]
        ref_latent_path = pth_st_et["latent_path"]
        st = pth_st_et["pred_start"]
        et = pth_st_et["pred_end"]
        # import pdb; pdb.set_trace()
        prompt = torch.load(ref_latent_path, map_location=model.device) # [b d t]
        prompt = prompt.transpose(1, 2) # [b t d]
        
        if st == -1:
            sf = 0
        else:
            sf = int(st * sampling_rate / downsample_rate )
        
        if et == -1:
            ef = max_frames
        else:
            ef = int(et * sampling_rate / downsample_rate )
        # prompt = prompt[:, :sf, :]
        return prompt, sf, ef
    else:
        prompt = torch.zeros(1, max_frames, 64).to(model.device)
        return prompt, 0, max_frames



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-config', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--lrc-path', type=str, default=None)
    parser.add_argument('--use-ref-latent', type=int, default=0)
    parser.add_argument('--ref-latent-path', type=str, default=0)
    parser.add_argument('--mulan-style-path', type=str, default=None)
    parser.add_argument('--infer-epoch', type=int, default=None)
    parser.add_argument('--cfg-strength', type=float, default=None)

    args = parser.parse_args()
    
    lrc_path = args.lrc_path
    use_ref_latent = args.use_ref_latent
    ref_style_path = args.mulan_style_path
    infer_epoch = args.infer_epoch
    cfg_strength = args.cfg_strength
    ref_latent_path = args.ref_latent_path
    
    with open(args.model_config) as f:
        model_config = json.load(f)

    model_cls = DiT
    ckpt_path = args.ckpt_path
    device='npu'
    use_style_prompt=True
    dit_model = CFM(
            transformer=model_cls(**model_config["model"], use_style_prompt=use_style_prompt),
            num_channels=model_config["model"]['mel_dim'],
            use_style_prompt=use_style_prompt
        )
    total_params = sum(p.numel() for p in dit_model.parameters())
    # import pdb; pdb.set_trace()
    print(f"Total parameters: {total_params}")
    text_enc_params = sum(p.numel() for p in dit_model.transformer.text_embed.parameters())
    # print(f"Parameters of text encoder: {text_enc_params}")
    dit_model = dit_model.to(device)
    dit_model = load_checkpoint(dit_model, ckpt_path, device=device, use_ema=True)

    # vae = torch.jit.load("/mnt/sfs/music/hkchen/workspace/F5-TTS-HW/exps/styleprompt/f5_tts/infer/vae.pt").to(device)

    t5_model_name = "t5-base"
    t5_model_path = "/mnt/sfs/music/hkchen/huggingface/t5_base"
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_path)
    t5_model = T5EncoderModel.from_pretrained(t5_model_path).to(device)
    
    lrc_tokenizer = CNENTokenizer()

    sampling_rate = 44100
    downsample_rate = 2048
    max_frames = 2048
    max_secs = max_frames / (sampling_rate / downsample_rate)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ckpt_step = os.path.splitext(os.path.basename(ckpt_path))[0].split('_')[-1]
    
    with open(lrc_path, 'r') as f:
        song_lyc = json.load(f)
        
    with open(ref_style_path, 'r') as f:
        song_name2style_ref_npy = json.load(f)
        
    with open(ref_latent_path, 'r') as f:
        song_name2ref_latent = json.load(f)        
    
    for epoch in tqdm(range(infer_epoch), total=infer_epoch):
        for song, lrc in tqdm(song_lyc.items()):
            lrc_prompt, start_time = get_lrc_prompt(lrc, lrc_tokenizer, dit_model, max_secs)
            latent_prompt, sf, ef = get_latent_prompt(dit_model, use_ref_latent, downsample_rate, song, max_frames, song_name2ref_latent)
            for style, pth in song_name2style_ref_npy[song].items():
                style_prompt = get_style_prompt(device=device, song_name=song, ref_npy_pth=pth)
                inference(model=dit_model,
                        cond=latent_prompt,
                        text=lrc_prompt,
                        duration=max_frames,
                        style_prompt=style_prompt,
                        style=style,
                        output_dir=output_dir,
                        song_name=song,
                        ckpt_step=ckpt_step,
                        start_time=start_time,
                        latent_pred_start_frame=sf,
                        latent_pred_end_frame=ef,
                        epoch=epoch,
                        cfg_strength=cfg_strength
                        )
