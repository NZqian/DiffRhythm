export PATH=/mnt/sfs/miniconda3/envs/310py/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$PWD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib

export ASCEND_RT_VISIBLE_DEVICES=7

# llama 1b fat
echo infer llama 1b fat align rand start add ste base model without bpm text mulan style emb from latent
python3 diffrhythm/infer/infer_custom_latent.py \
    --model-config diffrhythm/config/F5-llama-1b-400text-fat.json \
    --ckpt-path ../ckpts/base_model_without_bpm_pure_music_mulan_style_emb_from_latent/model_4750000.pt \
    --output-dir infer/base_model_without_bpm_pure_music_text_mulan_style_emb_from_latent \
    --lrc-path /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/song_stlye_lrc_without_bpm.json \
    --use-ref-latent 0 \
    --mulan-style-path /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/song_name2mulan_style_text.json \
    --ref-latent-path /mnt/sfs/music/hkchen/workspace/F5-TTS-HW/song_name2ref_latent_v2.json \
    --infer-epoch 3 \
    --cfg-strength 4.0
