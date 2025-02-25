export PATH=/mnt/sfs/miniconda3/envs/310py/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$PWD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib

export ASCEND_RT_VISIBLE_DEVICES=7

# llama 1b fat
echo infer llama 1b fat align rand start add ste base model without bpm text mulan style emb from latent
python3 /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/exps/base_model_without_bpm_pure_music_mulan_style_emb/f5_tts/infer/infer_backup.py \
    --model-config /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/exps/styleprompt_f5_llama/f5_tts/config/F5-llama-1b-400text-fat.json \
    --ckpt-path /mnt/sfs/music/zqning/F5-TTS/ckpts/dpo_gradckpt1_8gpu_bs16_cfm1_re/model_4901400.pt \
    --output-dir /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/exps/infer/base_model_without_bpm_pure_music_audio_mulan_style_emb_from_latent_asred \
    --lrc-path /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/song_stlye_lrc_without_bpm.json \
    --use-ref-latent 0 \
    --mulan-style-path /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/song_name2mulan_style_audio.json \
    --ref-latent-path /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/song_name2ref_latent_v2.json \
    --infer-epoch 1 \
    --cfg-strength 4.0


# echo infer llama 1b fat align rand start add ste base model without bpm audio mulan style emb from latent
# python3 /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/exps/base_model_without_bpm_pure_music_mulan_style_emb/f5_tts/infer/infer_custom_latent.py \
#     --model-config /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/exps/styleprompt_f5_llama/f5_tts/config/F5-llama-1b-400text-fat.json \
#     --ckpt-path /mnt/sfs/music/zqning/F5-TTS/ckpts/dpo_gradckpt1_8gpu_bs16_cfm1/model_4900200.pt \
#     --output-dir /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/exps/infer/base_model_without_bpm_pure_music_audio_mulan_style_emb_from_latent_asred_dpo \
#     --lrc-path /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/song_style_lyc_without_bpm_mulan.json \
#     --use-ref-latent 0 \
#     --mulan-style-path /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/song_name2mulan_style_audio_v2.json \
#     --ref-latent-path /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/song_name2ref_latent_v2.json \
#     --infer-epoch 1 \
#     --cfg-strength 4.0