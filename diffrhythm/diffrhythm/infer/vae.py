import torch
import torch_npu
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
import time
from tqdm import tqdm
import os
import argparse

device = "npu"

# Download model
sample_rate = 44100

config_path = "/mnt/sfs/music/hkchen/workspace/F5-TTS-HW/diffrhythm/diffrhythm/infer/stable_audio_tools/configs/vae_config.json"
ckpt_path = "/mnt/sfs/music/vae.ckpt"
model, model_config = get_pretrained_model(config_path, ckpt_path)
model = model.to(device)

io_channels = 2

def infer_single_dir(input_dir, output_dir):
    ROOT_DIR = input_dir
    paths = os.listdir(ROOT_DIR)
    paths = [os.path.join(ROOT_DIR, i) for i in paths if i.endswith('.pt')]

    with torch.no_grad():
        for pth in tqdm(paths):
            latent = torch.load(pth, map_location='npu')
            latent = latent.to(torch.float32) # [b d t]
            # import pdb; pdb.set_trace()
            if latent.shape[1] == 128:
                vocal_latent, accom_latent = torch.split(latent, 64, dim=1)
                vocal_output = model.decode_audio(vocal_latent, chunked=True)
                accom_output = model.decode_audio(accom_latent, chunked=True)
                
                vocal_output = rearrange(vocal_output, "b d n -> d (b n)")
                vocal_output = vocal_output.to(torch.float32).div(torch.max(torch.abs(vocal_output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                basename = os.path.basename(pth.replace('.pt', '_vocal.wav'))
                output_pth = os.path.join(output_dir, basename)
                torchaudio.save(output_pth, vocal_output, sample_rate)
                
                accom_output = rearrange(accom_output, "b d n -> d (b n)")
                accom_output = accom_output.to(torch.float32).div(torch.max(torch.abs(accom_output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                basename = os.path.basename(pth.replace('.pt', '_accom.wav'))
                output_pth = os.path.join(output_dir, basename)
                torchaudio.save(output_pth, accom_output, sample_rate)
                
                merged_output = vocal_output + accom_output
                basename = os.path.basename(pth.replace('.pt', '_merged.wav'))
                output_pth = os.path.join(output_dir, basename)
                torchaudio.save(output_pth, merged_output, sample_rate)
            else:
                if latent.shape[-1] == 6144:
                    latent_part1, latent_part2, latent_part3 = torch.chunk(latent, 3, -1)
                    output1 = model.decode_audio(latent_part1, chunked=True)
                    output2 = model.decode_audio(latent_part2, chunked=True)
                    output3 = model.decode_audio(latent_part3, chunked=True)
                    output = torch.cat((output1, output2, output3), dim=-1)
                else:
                    output = model.decode_audio(latent, chunked=True)

                # Rearrange audio batch to a single sequence
                output = rearrange(output, "b d n -> d (b n)")

                # Peak normalize, clip, convert to int16, and save to file
                output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                
                basename = os.path.basename(pth.replace('.pt', '.wav'))
                output_pth = os.path.join(output_dir, basename)
                torchaudio.save(output_pth, output, sample_rate)


def infer_multi_dir(input_dir, output_dir):
    ROOT_DIR = input_dir
    dirs = os.listdir(ROOT_DIR)
    dirs = [os.path.join(ROOT_DIR, i) for i in dirs]
    
    # import pdb; pdb.set_trace()
    
    for dir in tqdm(dirs, desc=f"prcessing dirs"):
        paths = os.listdir(dir)
        paths = [os.path.join(dir, i) for i in paths if i.endswith('.pt')]

        sub_dir = os.path.join(output_dir, os.path.basename(dir))
        os.makedirs(sub_dir, exist_ok=True)

        with torch.no_grad():
            for pth in tqdm(paths):
                latent = torch.load(pth, map_location='npu')
                latent = latent.to(torch.float32) # [b d t]
                # import pdb; pdb.set_trace()
                if latent.shape[1] == 128:
                    vocal_latent, accom_latent = torch.split(latent, 64, dim=1)
                    vocal_output = model.decode_audio(vocal_latent, chunked=True)
                    accom_output = model.decode_audio(accom_latent, chunked=True)
                    
                    vocal_output = rearrange(vocal_output, "b d n -> d (b n)")
                    vocal_output = vocal_output.to(torch.float32).div(torch.max(torch.abs(vocal_output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                    basename = os.path.basename(pth.replace('.pt', '_vocal.wav'))
                    output_pth = os.path.join(sub_dir, basename)
                    torchaudio.save(output_pth, vocal_output, sample_rate)
                    
                    accom_output = rearrange(accom_output, "b d n -> d (b n)")
                    accom_output = accom_output.to(torch.float32).div(torch.max(torch.abs(accom_output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                    basename = os.path.basename(pth.replace('.pt', '_accom.wav'))
                    output_pth = os.path.join(sub_dir, basename)
                    torchaudio.save(output_pth, accom_output, sample_rate)
                    
                    merged_output = vocal_output + accom_output
                    basename = os.path.basename(pth.replace('.pt', '_merged.wav'))
                    output_pth = os.path.join(sub_dir, basename)
                    torchaudio.save(output_pth, merged_output, sample_rate)
                else:
                    output = model.decode_audio(latent, chunked=True)
                    # latent = latent.unsqueeze(0)
                    # import pdb; pdb.set_trace()
                    # latent = latent.transpose(1, 2)
                    # output = model.decode_audio(vocal_latent, chunked=True)
                    # output = model.decode_export(latent)

                    # Rearrange audio batch to a single sequence
                    output = rearrange(output, "b d n -> d (b n)")

                    # Peak normalize, clip, convert to int16, and save to file
                    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                    
                    basename = os.path.basename(pth.replace('.pt', '.wav'))
                    output_pth = os.path.join(sub_dir, basename)
                    torchaudio.save(output_pth, output, sample_rate)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-m', '--mode', required=True)
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    mode = args.mode
    
    print(f"Infer {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if mode == "single":
        infer_single_dir(input_dir, output_dir)
    else:
        infer_multi_dir(input_dir, output_dir)