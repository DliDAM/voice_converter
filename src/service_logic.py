import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import streamlit as st

def init(progress_bar):    
    ckpt_converter = 'C:/Users/Admin/Documents/grad_project/voice_converter/checkpoints_v2/converter' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = 'C:/Users/Admin/Documents/grad_project/voice_converter/outputs_v2'
    
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    st.session_state.output_dir = output_dir
    st.session_state.tone_color_converter = tone_color_converter
    progress_bar.progress(50)

    speed = 1.0
    language = "KR"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TTS(language=language, device=device)
    st.session_state.model = model
    progress_bar.progress(100)

    os.makedirs(output_dir, exist_ok=True)

    #return model, output_dir, tone_color_converter

def get_se_from_audio_path(progress_bar, audio_path, tone_color_converter, output_dir):
    #reference_speaker = 'resources/openvoice_2.mp3' # 클론할 목소리가 들어갈 자리
    progress_bar.progress(50)
    target_se, audio_name = se_extractor.get_se(audio_path, tone_color_converter, vad=False)
    progress_bar.progress(100)
    src_path = f'{output_dir}/tmp.wav'
    return target_se, src_path

def inference(prompt, src_path, output_dir, tone_color_converter, target_se):
    
    speed = 1.0
    # language = "KR"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = TTS(language=language, device=device)
    model = st.session_state.model
    speaker_ids = model.hps.data.spk2id
    
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        model.tts_to_file(prompt, speaker_id, src_path, speed=speed)
        save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
        
    return save_path
    
    
    
    