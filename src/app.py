from service_logic import init, get_se_from_audio_path, inference
import streamlit as st
import os

# Function to display the main page
def main_page():
    #col1, col2, col3 = st.columns([1, 2, 1])
    #with col2:
        #st.image("",use_column_width = True)# "" 안에 이미지 파일
        #st.image("")
    st.write("""
            들리담 실시간 음성 생성 및 변조 데모\n
            방병훈, 이한결, 김도연, 원윤서, 박소은""")    
    if st.button("데모 시작"):
        st.write("모델 로드 중")
        progress_bar = st.progress(0)
        output_dir, tone_color_converter = init() 
        st.success("모델 로드 완료")
        st.session_state.page = 2
    return output_dir, tone_color_converter

# Function to display a question page
def demo_page(output_dir, tone_color_converter):
    st.set_page_config(layout="wide")
    
    uploaded_file = st.file_uploader("변조에 이용할 레퍼런스 목소리 파일을 넣어주세요.", type=["mp3"])
    if uploaded_file is not None:
        save_directory = "C:/Users/Admin/Documents/grad_project/voice_converter/resources"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        audio_path = os.path.join(save_directory, uploaded_file.name)
    
        # 파일을 지정된 경로에 저장
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    target_se, src_path = get_se_from_audio_path(audio_path, tone_color_converter, output_dir)
    # 필요하면 위 과정 동안 버퍼링 생성
    #st.write("")
    prompt = st.text_area("생성할 문장을 입력해주세요")
    save_path = inference(prompt, src_path, output_dir, tone_color_converter, target_se)
    if st.button("재생하기"):
        st.audio(save_path, format='audio/mp3') 

# Function to display the final results page
def final_page():
    st.set_page_config(layout="wide")
    st.write("# 최종 결과")
    st.write("결과 내용 표시")  # Add the actual result content

# Initialize session state
if 'main' not in st.session_state:
    st.session_state = 'main'

# Page navigation
while st.session_state.page == 1:
    output_dir, tone_color_converter = main_page()

while st.session_state.page == 2:
    demo_page(output_dir, tone_color_converter)


