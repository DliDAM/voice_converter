from src.service_logic import init, get_se_from_audio_path, inference
import streamlit as st
import os

# 초기 페이지 설정
if 'page' not in st.session_state:
    st.session_state.page = "main_page"

# 음성 조절 함수 (예제 함수로 실제 음성 조절 기능은 구현되지 않음)
def adjust_voice(pitch, rate, volume, echo, reverb, speed, tone):
    st.write(f"Pitch: {pitch}")
    st.write(f"Rate: {rate}")
    st.write(f"Volume: {volume}")
    st.write(f"Echo: {echo}")
    st.write(f"Reverb: {reverb}")
    st.write(f"Speed: {speed}")
    st.write(f"Tone: {tone}")
    st.write("음성 파라미터가 조정되었습니다.")

# 메인 페이지 함수
def main_page():
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.markdown("# 들리담")
    st.markdown("## 들리담 설명")
    st.markdown("들리담 설명")
    st.markdown("들리담 실시간 음성 생성 및 변조 데모\n방병훈, 이한결, 김도연, 원윤서, 박소은")
    if st.button("데모 실행하기"):
        
        st.write("모델 로드 중")
        progress_bar = st.progress(0)
        #output_dir, tone_color_converter = init() 
        init(progress_bar)
        st.success("모델 로드 완료")
        # st.session_state.output_dir = output_dir
        # st.session_state.tone_color_converter = tone_color_converter
        st.session_state.page = "execution_page"
        st.experimental_rerun()

# 실행 페이지 함수
def execution_page():
    st.title("실행 페이지")
    st.write("여기는 실행 페이지입니다.")
    st.write("이 페이지에서 음성 파라미터를 조정할 수 있습니다.")
    
    # 사이드바에 슬라이더 추가
    pitch = st.sidebar.slider("Pitch", 0.5, 2.0, 1.0, 0.1)
    rate = st.sidebar.slider("Rate", 0.5, 2.0, 1.0, 0.1)
    volume = st.sidebar.slider("Volume", 0.0, 1.0, 0.5, 0.1)
    echo = st.sidebar.slider("Echo", 0.0, 1.0, 0.0, 0.1)
    reverb = st.sidebar.slider("Reverb", 0.0, 1.0, 0.0, 0.1)
    speed = st.sidebar.slider("Speed", 0.5, 2.0, 1.0, 0.1)
    tone = st.sidebar.slider("Tone", 0.5, 2.0, 1.0, 0.1)
    
    uploaded_file = st.file_uploader("변조에 이용할 레퍼런스 목소리 파일을 넣어주세요.", type=["mp3"])
    if uploaded_file is not None:
        save_directory = "C:/Users/Admin/Documents/grad_project/voice_converter/resources"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        audio_path = os.path.join(save_directory, uploaded_file.name)
    
        # 파일을 지정된 경로에 저장
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("음성 세그먼트 생성중")
        progress_bar = st.progress(0)
        target_se, src_path = get_se_from_audio_path(progress_bar,audio_path, st.session_state.tone_color_converter, st.session_state.output_dir)
        
        prompt = st.text_area("생성할 문장을 입력해주세요")
        save_path = inference(prompt, src_path, st.session_state.output_dir, st.session_state.tone_color_converter, target_se)
    
    if st.button("재생하기"):
        st.audio(save_path, format='audio/mp3') 

    # 사용자가 파라미터 조정 후 버튼을 누르면 함수 실행
    if st.sidebar.button("Apply Voice Parameters"):
        adjust_voice(pitch, rate, volume, echo, reverb, speed, tone)
    
    if st.button("메인 페이지로 돌아가기"):
        st.session_state.page = "main_page"
        st.experimental_rerun()

# 현재 페이지에 따라 다른 함수 실행
if st.session_state.page == "main_page":
    main_page()
else:
    execution_page()
