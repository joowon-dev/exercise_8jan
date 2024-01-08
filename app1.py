import numpy as np
import pickle
import pandas as pd
import streamlit as st 

# 두 모델 로드 (실제 모델 파일 이름으로 'reg_RF.pkl' 및 'other_model.pkl' 교체 필요)
pickle_in_rf = open("reg_RF.pkl", "rb")
rf_model = pickle.load(pickle_in_rf)

pickle_in_other = open("other_model.pkl", "rb")
other_model = pickle.load(pickle_in_other)

def predict_std_perf(model, hour_s, pre_score):
    prediction = model.predict([[hour_s, pre_score]])
    print(prediction)
    return prediction

def main():
    st.title("학생 성적 예측 시스템")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit 학생 성적 예측기 </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    hour_s = st.text_input("공부 시간", "입력하세요")
    pre_score = st.text_input("이전 성적", "입력하세요")

    # 모델 선택 체크박스
    use_rf_model = st.checkbox('랜덤 포레스트 모델 사용', value=True)
    use_other_model = st.checkbox('다른 모델 사용')

    result = ""
    if st.button("예측"):
        if use_rf_model:
            result = predict_std_perf(rf_model, hour_s, pre_score)
        elif use_other_model:
            result = predict_std_perf(other_model, hour_s, pre_score)

    st.success('결과는 {}입니다'.format(result))

    if st.button("정보"):
        st.text("파이썬으로 배워봅시다")
        st.text("Streamlit으로 구축됨")

if __name__ == '__main__':
    main()
