export ASCEND_RT_VISIBLE_DEVICES=7

export PATH=/mnt/sfs/miniconda3/envs/310py/bin:$PATH

INPUT_DIR="/mnt/sfs/music/hkchen/workspace/F5-TTS-HW/exps/infer/base_model_without_bpm_pure_music_audio_mulan_style_emb_from_latent"
OUTPUT_DIR="/mnt/sfs/music/hkchen/workspace/F5-TTS-HW/vae_test"
MODE="single"


python vae.py --input-dir ${INPUT_DIR} \
                --output-dir ${OUTPUT_DIR} \
                --mode ${MODE}