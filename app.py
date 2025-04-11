# --- START OF FILE app.py ---

import streamlit as st
import os
from PIL import Image, UnidentifiedImageError
import io
from datetime import datetime
from streamlit_option_menu import option_menu # 상단 메뉴 UI
from streamlit_image_comparison import image_comparison # 이미지 비교 컴포넌트
import openai # OpenAI 라이브러리 추가

# --- 로컬 유틸리티 및 스타일 함수 임포트 ---
from utils import (
    load_image, apply_makeup, apply_fashion_filter, virtual_try_on,
    create_assets_folder, detect_face_landmarks, change_clothing_color,
    apply_makeup_transfer # 메이크업 전송 함수 추가
)
from style_transfer import (
    prepare_clothing_samples, prepare_makeup_style_samples, MAKEUP_STYLES_INFO
)

# --- OpenAI API Key 설정 (Streamlit Secrets 사용) ---
# 로컬 테스트 시: secrets.toml 파일에 OPENAI_API_KEY = "sk-..." 형식으로 저장
# 배포 시: Streamlit Community Cloud의 Secrets 설정에 추가
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ OpenAI API Key를 찾을 수 없습니다. .streamlit/secrets.toml 파일을 확인하거나 Streamlit Cloud Secrets 설정을 확인하세요.")
    st.stop() # API 키 없으면 앱 중단
except KeyError:
    st.error("⚠️ secrets.toml 파일이나 Streamlit Cloud Secrets에 'OPENAI_API_KEY' 항목이 없습니다.")
    st.stop()


# --- ⚙️ 앱 설정 및 초기화 ⚙️ ---
st.set_page_config(
    page_title="✨ AI 스타일리스트 ✨ - Pro",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed", # 사이드바 기본 닫힘
)

# --- 🎨 CSS 스타일 적용 (선택적) ---
# (이전 CSS 코드와 동일)
st.markdown("""
<style>
    /* ... (이전 CSS 코드 생략) ... */
</style>
""", unsafe_allow_html=True)


# --- 📁 경로 및 상수 정의 ---
ASSETS_DIR = "assets"
EXAMPLES_DIR = os.path.join(ASSETS_DIR, "examples")
CLOTHES_DIR = os.path.join(ASSETS_DIR, "clothes")
MAKEUP_STYLES_DIR = os.path.join(ASSETS_DIR, "makeup_styles")

# --- 🚀 앱 시작 시 초기화 작업 ---
create_assets_folder()

# --- 🖼️ 리소스 로드 (캐싱 활용) ---
@st.cache_resource
def load_resources():
    print("Loading resources...")
    resources = {
        "examples": {},
        "clothing": prepare_clothing_samples(use_local=True, local_dir=CLOTHES_DIR),
        "makeup_styles": prepare_makeup_style_samples(local_dir=MAKEUP_STYLES_DIR)
    }
    # 예제 이미지 로드
    if os.path.isdir(EXAMPLES_DIR):
        try:
            example_filenames = [f for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for f in example_filenames:
                 name = os.path.splitext(f)[0].replace("_", " ").title()
                 path = os.path.join(EXAMPLES_DIR, f)
                 resources["examples"][name] = path # 경로만 저장, 로드는 필요 시
        except Exception as e:
            print(f"Error loading example image list: {e}")
    print("Resources loaded.")
    return resources

RESOURCES = load_resources()
AVAILABLE_CLOTHING_TYPES = list(RESOURCES["clothing"].keys())
AVAILABLE_MAKEUP_STYLES = list(RESOURCES["makeup_styles"].keys())
AVAILABLE_EXAMPLE_NAMES = ["이미지 업로드"] + list(RESOURCES["examples"].keys())
AVAILABLE_FASHION_STYLES = ["casual", "vintage", "elegant", "monochrome"] # Define available fashion styles


# --- 📌 세션 상태 관리 ---
# 세션 상태 초기화 함수
def initialize_session_state():
    # 앱 모드 선택
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "홈"
    # 업로드/선택된 원본 이미지
    if "original_image" not in st.session_state:
        st.session_state.original_image = None
    if "original_image_caption" not in st.session_state:
        st.session_state.original_image_caption = ""
    # 각 모드별 결과 이미지
    if "filtered_image" not in st.session_state:
        st.session_state.filtered_image = None
    if "makeup_image" not in st.session_state:
        st.session_state.makeup_image = None
    if "tryon_image" not in st.session_state:
        st.session_state.tryon_image = None
    # 현재 적용된 결과 캡션
    if "result_caption" not in st.session_state:
        st.session_state.result_caption = ""
    # 갤러리
    if "gallery" not in st.session_state:
        st.session_state.gallery = [] # {'image': PIL Image, 'caption': str} list
    # 메이크업 옵션
    if "makeup_options" not in st.session_state:
        st.session_state.makeup_options = {
            'intensity': 0.6, 'apply_lips': True, 'lip_color': '#E64E6B', 'lip_intensity': 0.7,
            'apply_eyeshadow': True, 'eyeshadow_color': '#8A5A94', 'eyeshadow_intensity': 0.5,
            'apply_blush': False, 'blush_color': '#F08080', 'blush_intensity': 0.4,
        }
    # 가상 피팅 옵션
    if "tryon_options" not in st.session_state:
        st.session_state.tryon_options = {
            'pos_x': 50, 'pos_y': 100, 'scale': 1.0, 'color_change': False, 'target_color': '#FF5733',
            'selected_clothing': None # 현재 선택된 의상 종류 저장
        }
    # AI 추천 관련 상태
    if "recommendation_prompt" not in st.session_state:
        st.session_state.recommendation_prompt = ""
    if "recommendation_result" not in st.session_state:
        st.session_state.recommendation_result = ""

initialize_session_state()

# --- ✨ 상단 네비게이션 메뉴 ---
# "AI 추천" 메뉴 추가
menu_options = ["홈", "패션 필터", "메이크업", "가상 피팅", "AI 추천", "갤러리"]
menu_icons = ['house-door-fill', 'palette-fill', 'magic', 'person-standing-dress', 'robot', 'images']

selected_mode = option_menu(
    menu_title=None,
    options=menu_options,
    icons=menu_icons,
    menu_icon="gem",
    default_index=menu_options.index(st.session_state.app_mode), # 현재 모드 선택 유지
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f2f6"},
        "icon": {"color": "#ff6347", "font-size": "20px"},
        "nav-link": {
            "font-size": "16px", "text-align": "center", "margin": "0px 5px", "--hover-color": "#e0e0e0",
            "padding": "10px 15px", "border-radius": "5px"
        },
        "nav-link-selected": {"background-color": "#F63366", "color": "white"}, # config.toml primaryColor
    }
)

# 선택된 모드를 세션 상태에 업데이트
if selected_mode:
    # 모드가 변경되면 이전 결과 이미지를 초기화하여 혼동 방지 (AI 추천 결과는 유지 가능)
    if st.session_state.app_mode != selected_mode and selected_mode != "AI 추천":
        st.session_state.filtered_image = None
        st.session_state.makeup_image = None
        st.session_state.tryon_image = None
        st.session_state.result_caption = ""
        # 추천 프롬프트/결과는 유지할지 초기화할지 선택 (여기서는 유지)
        # st.session_state.recommendation_prompt = ""
        # st.session_state.recommendation_result = ""
    st.session_state.app_mode = selected_mode


# --- 🖼️ 이미지 업로드 및 선택 (사이드바 사용 최소화, 필요 시 확장 패널 사용) ---
# (이전 이미지 업로드/선택 코드와 동일)
# ... (코드 생략) ...
with st.expander("📂 이미지 선택 및 업로드", expanded=(st.session_state.original_image is None)): # 이미지가 없으면 기본 확장
    image_source = st.selectbox("이미지 소스 선택:", AVAILABLE_EXAMPLE_NAMES, index=0, key="image_source_select")

    if image_source == "이미지 업로드":
        uploaded_file = st.file_uploader(
            "여기에 이미지 파일을 드래그하거나 클릭하여 업로드하세요.",
            type=["jpg", "jpeg", "png"],
            key="file_uploader"
        )
        if uploaded_file is not None:
            try:
                # 이미지 로드 및 세션 상태 업데이트
                loaded_image = load_image(uploaded_file)
                if loaded_image:
                     # 새 이미지 로드 시 현재 이미지와 다른 경우에만 업데이트 (중복 로딩 방지)
                    current_caption = f"업로드: {uploaded_file.name}"
                    if st.session_state.original_image is None or st.session_state.original_image_caption != current_caption:
                        st.session_state.original_image = loaded_image
                        st.session_state.original_image_caption = current_caption
                        # 새 이미지 로드 시 이전 결과 초기화
                        st.session_state.filtered_image = None
                        st.session_state.makeup_image = None
                        st.session_state.tryon_image = None
                        st.session_state.result_caption = ""
                        st.success("✅ 이미지가 성공적으로 업로드되었습니다.")
                        # st.image(st.session_state.original_image, caption="업로드된 원본 이미지", width=300) # Expander 내부에서는 생략 가능
                        st.rerun() # 새 이미지 로드 후 UI 즉시 갱신
                else:
                    st.error("이미지 로드에 실패했습니다.")
                    st.session_state.original_image = None
            except Exception as e:
                st.error(f"이미지 처리 오류: {e}")
                st.session_state.original_image = None
    else: # 예제 이미지 선택
        example_path = RESOURCES["examples"].get(image_source)
        if example_path and os.path.exists(example_path):
            try:
                # 예제 로드 및 세션 상태 업데이트
                current_caption = f"예제: {image_source}"
                # 현재 이미지와 다를 경우에만 업데이트
                if st.session_state.original_image is None or st.session_state.original_image_caption != current_caption:
                    loaded_image = load_image(example_path)
                    if loaded_image:
                        st.session_state.original_image = loaded_image
                        st.session_state.original_image_caption = current_caption
                        st.session_state.filtered_image = None
                        st.session_state.makeup_image = None
                        st.session_state.tryon_image = None
                        st.session_state.result_caption = ""
                        st.success(f"✅ '{image_source}' 예제 이미지가 선택되었습니다.")
                        # st.image(st.session_state.original_image, caption="선택된 원본 이미지", width=300) # Expander 내부에서는 생략 가능
                        st.rerun() # 새 이미지 로드 후 UI 즉시 갱신
                    else:
                        st.error(f"'{image_source}' 예제 이미지 로드 실패.")
                        st.session_state.original_image = None
            except Exception as e:
                st.error(f"예제 이미지 처리 오류: {e}")
                st.session_state.original_image = None


# --- 🤖 GPT 기반 추천 함수 ---
def get_style_recommendation(user_prompt):
    """GPT API를 호출하여 스타일 추천을 받는 함수"""
    if not user_prompt:
        return "추천을 받으려면 설명을 입력해주세요."

    # 시스템 메시지: GPT의 역할과 응답 방향 정의
    system_message = f"""
    당신은 전문 패션 & 뷰티 AI 스타일리스트입니다. 사용자의 설명(피부톤, 선호 스타일, 상황 등)을 바탕으로 구체적이고 실용적인 스타일링 추천을 제공해주세요.
    추천 내용은 다음 요소들을 포함할 수 있습니다:
    - 메이크업: 어울리는 립, 아이섀도우, 블러셔 색상 및 스타일 (예: '쿨톤 피부에는 핑크 계열 립스틱과 회갈색 아이섀도우가 잘 어울립니다.')
    - 패션 필터: 사용자가 앱에서 선택할 수 있는 필터 추천 ({', '.join(AVAILABLE_FASHION_STYLES)}) (예: '차분한 느낌을 원하시면 'elegant' 필터를 적용해보세요.')
    - 의상 종류/색상: 사용자가 앱에서 선택할 수 있는 의상 종류 ({', '.join(AVAILABLE_CLOTHING_TYPES)}) 또는 일반적인 의상 색상 추천 (예: '중요한 자리라면 'formal_dress'를 시도해보거나, 네이비 색상의 블라우스가 좋습니다.')
    - 추천 이유를 간략하게 설명해주세요.
    - 응답은 친절하고 이해하기 쉬운 한국어로 작성해주세요.
    - 마크다운 형식을 활용하여 가독성을 높여주세요 (예: 항목별 리스트 사용).
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # 또는 "gpt-4" 등 사용 가능한 모델
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7, # 약간의 창의성 허용
            max_tokens=300 # 응답 길이 제한
        )
        recommendation = response.choices[0].message.content.strip()
        return recommendation
    except openai.AuthenticationError:
        st.error("❌ OpenAI API 인증 오류: API 키가 유효하지 않거나 설정되지 않았습니다.")
        return None # 오류 발생 시 None 반환
    except openai.RateLimitError:
        st.error("❌ OpenAI API 호출 한도 초과: 잠시 후 다시 시도해주세요.")
        return None
    except Exception as e:
        st.error(f"❌ GPT API 호출 중 오류 발생: {e}")
        return None


# --- 📌 메인 콘텐츠 영역 ---
if st.session_state.app_mode == "홈":
    # (이전 홈 코드와 동일)
    st.header("💎 AI 스타일리스트 Pro에 오신 것을 환영합니다!")
    st.write("""
    위의 '이미지 선택 및 업로드' 섹션에서 이미지를 선택하거나 업로드하세요.
    그런 다음, 상단 메뉴에서 원하는 스타일링 기능을 선택하여 가상 체험을 즐겨보세요!
    새로운 **AI 추천** 메뉴에서 개인 맞춤 스타일링 조언을 받아볼 수도 있습니다.
    """)
    st.info("✨ **Tip:** 각 기능별로 다양한 옵션을 조절하여 자신만의 스타일을 만들어보세요!")
    if st.session_state.original_image:
         st.success("✅ 원본 이미지가 준비되었습니다. 상단 메뉴를 이용해 스타일링을 시작하거나 AI 추천을 받아보세요!")
         st.image(st.session_state.original_image, caption=st.session_state.original_image_caption, width=500)


elif st.session_state.app_mode == "갤러리":
    # (이전 갤러리 코드와 동일)
    # ... (코드 생략) ...
    st.header("🖼️ 나의 스타일 갤러리")
    if not st.session_state.gallery:
        st.info("아직 갤러리에 저장된 이미지가 없습니다. 스타일 적용 후 결과 하단의 '갤러리에 저장' 버튼을 눌러 추가해보세요.")
    else:
        st.success(f"총 {len(st.session_state.gallery)}개의 스타일 이미지가 저장되어 있습니다.")
        cols = st.columns(4) # 4열로 표시
        for i, img_data in enumerate(st.session_state.gallery):
            img = img_data['image']
            caption = img_data['caption']
            with cols[i % 4]:
                st.image(img, caption=f"{i+1}: {caption}", use_container_width=True)
                buf = io.BytesIO()
                save_img_gal = img.convert('RGB') if img.mode == 'RGBA' else img
                try:
                    save_img_gal.save(buf, format="PNG")
                    st.download_button(
                        label="💾", # 아이콘 형태 버튼
                        data=buf.getvalue(),
                        file_name=f"gallery_{caption.replace(' ', '_').replace(':', '_').replace('/', '_')}_{i}.png", # 파일명 유효 문자 처리
                        mime="image/png",
                        key=f"gallery_download_{i}",
                        use_container_width=True,
                        help="다운로드"
                    )
                except Exception as e:
                    st.error(f"이미지 저장 오류: {e}")
        st.divider()
        if st.button("🗑️ 갤러리 모두 비우기", use_container_width=True, type="primary"):
            st.session_state.gallery = []
            st.success("갤러리가 비워졌습니다.")
            st.rerun()

# --- ✨ AI 추천 모드 ---
elif st.session_state.app_mode == "AI 추천":
    st.header("🤖 AI 스타일 추천")
    st.write("""
    자신의 특징(예: 피부톤, 헤어 컬러), 선호하는 스타일, 또는 스타일링이 필요한 상황(TPO)을 설명해주세요.
    AI가 설명을 바탕으로 이 앱의 기능(메이크업 색상, 패션 필터, 가상 의상 등)과 관련된 맞춤 스타일링 팁을 제안합니다.
    """)

    # 사용자 입력
    placeholder_text = "예시:\n- 저는 가을 웜톤이고, 평소 내추럴 메이크업을 선호해요. 데일리룩 추천해주세요.\n- 중요한 면접이 있는데 어떤 스타일이 좋을까요? 신뢰감을 주는 인상을 원해요.\n- 여름 휴가 때 해변에서 입을 만한 밝은 색 옷 추천해주세요!"
    st.session_state.recommendation_prompt = st.text_area(
        "스타일링 고민이나 원하는 내용을 설명해주세요:",
        value=st.session_state.recommendation_prompt,
        placeholder=placeholder_text,
        height=150,
        key="recommend_input"
    )

    # 추천 받기 버튼
    if st.button("✨ AI에게 추천 받기", key="get_recommendation", type="primary", use_container_width=True):
        if st.session_state.recommendation_prompt:
            with st.spinner("🧠 AI가 열심히 추천 내용을 생성 중입니다..."):
                recommendation = get_style_recommendation(st.session_state.recommendation_prompt)
                if recommendation: # 오류가 아닐 경우에만 결과 업데이트
                    st.session_state.recommendation_result = recommendation
                # 오류 발생 시에는 get_style_recommendation 함수 내에서 st.error로 메시지 표시됨
        else:
            st.warning("⚠️ 추천을 받으려면 설명을 입력해주세요.")

    # 추천 결과 표시
    if st.session_state.recommendation_result:
        st.divider()
        st.subheader("💡 AI 추천 결과")
        st.markdown(st.session_state.recommendation_result)


# --- 이미지 입력이 필요한 모드 ---
elif st.session_state.original_image is None:
    st.warning("⚠️ 먼저 '이미지 선택 및 업로드'에서 이미지를 선택하거나 업로드해주세요.")
    # 여기서 실행 중지 (이후 코드는 original_image가 있다고 가정함)

else: # 원본 이미지가 로드된 경우에만 실행

    # --- ✨ 패션 필터 모드 ---
    if st.session_state.app_mode == "패션 필터":
        # (이전 패션 필터 코드와 동일)
        # ... (코드 생략) ...
        st.header("🎨 패션 스타일 필터")
        col1, col2 = st.columns([1, 3]) # 옵션 영역 / 결과 영역

        with col1: # 옵션 설정
            st.subheader("필터 옵션")
            selected_style = st.selectbox("스타일 선택:", ["선택 안함"] + AVAILABLE_FASHION_STYLES, key="filter_style")
            intensity = st.slider("효과 강도:", 0.0, 1.0, 0.7, 0.05, key="filter_intensity", help="0.0은 원본, 1.0은 최대 효과")
            apply_filter_btn = st.button("✨ 필터 적용", key="apply_filter", use_container_width=True, type="primary", disabled=(selected_style=="선택 안함"))

            if apply_filter_btn:
                with st.spinner("🎨 필터 적용 중..."):
                    try:
                        st.session_state.filtered_image = apply_fashion_filter(st.session_state.original_image, selected_style, intensity)
                        st.session_state.result_caption = f"{selected_style} 필터 (강도: {intensity:.2f})"
                        st.success("✅ 필터 적용 완료!")
                    except Exception as e:
                        st.error(f"필터 적용 중 오류 발생: {e}")
                        st.session_state.filtered_image = None # 오류 시 결과 초기화

        with col2: # 결과 표시
            st.subheader("결과 미리보기")
            if st.session_state.filtered_image:
                image_comparison(
                    img1=st.session_state.original_image,
                    img2=st.session_state.filtered_image,
                    label1="원본",
                    label2=st.session_state.result_caption,
                    width=700, # 비교 컴포넌트 너비
                    starting_position=50,
                    show_labels=True
                )
                # 결과 저장 버튼
                st.divider()
                save_col1, save_col2 = st.columns(2)
                with save_col1:
                    try:
                        buf = io.BytesIO()
                        save_img_fil = st.session_state.filtered_image.convert('RGB') if st.session_state.filtered_image.mode == 'RGBA' else st.session_state.filtered_image
                        save_img_fil.save(buf, format="PNG")
                        st.download_button("💾 결과 다운로드", buf.getvalue(), f"filter_{st.session_state.result_caption.replace(' ', '_').replace(':', '_').replace('/', '_')}.png", "image/png", use_container_width=True)
                    except Exception as e:
                        st.error(f"파일 저장 준비 중 오류: {e}")
                with save_col2:
                     if st.button("🖼️ 갤러리에 저장", key="save_filter_gallery", use_container_width=True):
                         st.session_state.gallery.append({'image': st.session_state.filtered_image, 'caption': st.session_state.result_caption})
                         st.success("갤러리에 저장됨!")
            elif apply_filter_btn: # 버튼은 눌렀지만 결과가 없을 때 (오류 발생 등)
                st.info("필터 적용 결과를 기다리거나 적용에 실패했습니다.")
                st.image(st.session_state.original_image, caption="원본 이미지", width=400)
            elif selected_style != "선택 안함":
                st.info("👈 '필터 적용' 버튼을 눌러 결과를 확인하세요.")
                st.image(st.session_state.original_image, caption="원본 이미지", width=400)
            else:
                 st.image(st.session_state.original_image, caption="원본 이미지", width=400)


    # --- 💄 메이크업 모드 ---
    elif st.session_state.app_mode == "메이크업":
        # (이전 메이크업 코드와 동일)
        # ... (코드 생략) ...
        st.header("💄 AI 가상 메이크업")
        tab_manual, tab_transfer = st.tabs(["🎨 직접 메이크업", "✨ 스타일 전송"])

        # --- 🎨 직접 메이크업 탭 ---
        with tab_manual:
            col1_mu, col2_mu = st.columns([1, 3]) # 옵션 영역 / 결과 영역

            with col1_mu: # 옵션 설정
                st.subheader("메이크업 옵션")
                # 각 메이크업 요소 적용 여부 및 세부 옵션
                st.session_state.makeup_options['intensity'] = st.slider("전체 강도", 0.1, 1.0, st.session_state.makeup_options['intensity'], 0.05, key="mu_intensity")
                st.divider()
                st.session_state.makeup_options['apply_lips'] = st.checkbox("💄 입술", value=st.session_state.makeup_options['apply_lips'], key="mu_apply_lips")
                if st.session_state.makeup_options['apply_lips']:
                    st.session_state.makeup_options['lip_color'] = st.color_picker('립 색상', st.session_state.makeup_options['lip_color'], key="mu_lip_color")
                    st.session_state.makeup_options['lip_intensity'] = st.slider("립 강도", 0.1, 1.0, st.session_state.makeup_options['lip_intensity'], 0.05, key="mu_lip_intensity")
                st.divider()
                st.session_state.makeup_options['apply_eyeshadow'] = st.checkbox("✨ 아이섀도우", value=st.session_state.makeup_options['apply_eyeshadow'], key="mu_apply_eyeshadow")
                if st.session_state.makeup_options['apply_eyeshadow']:
                    st.session_state.makeup_options['eyeshadow_color'] = st.color_picker('섀도우 색상', st.session_state.makeup_options['eyeshadow_color'], key="mu_eyeshadow_color")
                    st.session_state.makeup_options['eyeshadow_intensity'] = st.slider("섀도우 강도", 0.1, 1.0, st.session_state.makeup_options['eyeshadow_intensity'], 0.05, key="mu_eyeshadow_intensity")
                st.divider()
                st.session_state.makeup_options['apply_blush'] = st.checkbox("😊 블러셔", value=st.session_state.makeup_options['apply_blush'], key="mu_apply_blush")
                if st.session_state.makeup_options['apply_blush']:
                     st.session_state.makeup_options['blush_color'] = st.color_picker('블러셔 색상', st.session_state.makeup_options['blush_color'], key="mu_blush_color")
                     st.session_state.makeup_options['blush_intensity'] = st.slider("블러셔 강도", 0.1, 1.0, st.session_state.makeup_options['blush_intensity'], 0.05, key="mu_blush_intensity")
                st.divider()
                apply_makeup_btn = st.button("💋 메이크업 적용", key="apply_makeup", use_container_width=True, type="primary",
                                             disabled=not (st.session_state.makeup_options['apply_lips'] or
                                                           st.session_state.makeup_options['apply_eyeshadow'] or
                                                           st.session_state.makeup_options['apply_blush']))

                if apply_makeup_btn:
                    with st.spinner("🧠 얼굴 분석 및 메이크업 적용 중..."):
                        try:
                            result_img, success = apply_makeup(st.session_state.original_image, st.session_state.makeup_options)
                            if success:
                                st.session_state.makeup_image = result_img
                                applied_list = [k.split('_')[1].capitalize() for k, v in st.session_state.makeup_options.items() if k.startswith('apply_') and v]
                                st.session_state.result_caption = f"직접 메이크업 ({', '.join(applied_list)})"
                                st.success("✅ 메이크업 적용 완료!")
                            else:
                                st.error("⚠️ 얼굴 감지 실패 또는 메이크업 적용에 문제가 발생했습니다.")
                                # Optionally keep the previous makeup image or reset it
                                # st.session_state.makeup_image = None
                        except Exception as e:
                             st.error(f"메이크업 적용 중 오류 발생: {e}")
                             st.session_state.makeup_image = None

            with col2_mu: # 결과 표시
                st.subheader("결과 미리보기 (직접)")
                # 결과 표시 조건 수정: 현재 모드가 '메이크업'이고, '직접 메이크업' 결과가 있을 때
                if st.session_state.makeup_image and st.session_state.result_caption.startswith("직접 메이크업"):
                    image_comparison(
                        img1=st.session_state.original_image,
                        img2=st.session_state.makeup_image,
                        label1="원본",
                        label2=st.session_state.result_caption,
                        width=700,
                        starting_position=50,
                        show_labels=True
                    )
                    st.divider()
                    save_col1, save_col2 = st.columns(2)
                    with save_col1:
                        try:
                            buf = io.BytesIO()
                            save_img_mu = st.session_state.makeup_image.convert('RGB') if st.session_state.makeup_image.mode == 'RGBA' else st.session_state.makeup_image
                            save_img_mu.save(buf, format="PNG")
                            st.download_button("💾 결과 다운로드", buf.getvalue(), f"makeup_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", "image/png", use_container_width=True)
                        except Exception as e:
                            st.error(f"파일 저장 준비 중 오류: {e}")
                    with save_col2:
                        if st.button("🖼️ 갤러리에 저장", key="save_makeup_gallery", use_container_width=True):
                            st.session_state.gallery.append({'image': st.session_state.makeup_image, 'caption': st.session_state.result_caption})
                            st.success("갤러리에 저장됨!")
                elif apply_makeup_btn: # 버튼 눌렀는데 아직 결과가 없다면 (오류 상황 등)
                    st.info("메이크업 결과를 기다리는 중이거나 적용에 실패했습니다.")
                    st.image(st.session_state.original_image, caption="원본 이미지", width=400)
                else:
                     st.info("👈 옵션을 선택하고 '메이크업 적용' 버튼을 누르세요.")
                     st.image(st.session_state.original_image, caption="원본 이미지", width=400)

        # --- ✨ 스타일 전송 탭 ---
        with tab_transfer:
            col1_tr, col2_tr = st.columns([1, 3])

            with col1_tr:
                st.subheader("메이크업 스타일 선택")
                if not AVAILABLE_MAKEUP_STYLES:
                    st.warning("사용 가능한 메이크업 스타일이 없습니다. `assets/makeup_styles` 폴더를 확인하세요.")
                    selected_style_name = None
                else:
                    selected_style_name = st.selectbox(
                        "참고할 스타일 선택:",
                        ["스타일 선택..."] + AVAILABLE_MAKEUP_STYLES,
                        key="makeup_style_select"
                    )

                style_image_pil = None
                if selected_style_name and selected_style_name != "스타일 선택...":
                    style_image_pil = RESOURCES["makeup_styles"].get(selected_style_name)
                    if style_image_pil:
                        st.image(style_image_pil, caption=f"선택된 스타일: {selected_style_name}", use_container_width=True)
                        # 스타일 설명 표시 (있으면)
                        style_info = MAKEUP_STYLES_INFO.get(selected_style_name)
                        if style_info:
                            st.caption(style_info.get('description', ''))
                    else:
                        st.error(f"'{selected_style_name}' 스타일 이미지를 로드할 수 없습니다.")

                apply_transfer_btn = st.button(
                    "✨ 스타일 적용하기",
                    key="apply_transfer",
                    use_container_width=True,
                    type="primary",
                    disabled=(selected_style_name is None or selected_style_name == "스타일 선택..." or style_image_pil is None)
                )

                if apply_transfer_btn and style_image_pil:
                    with st.spinner("🎨 스타일 분석 및 메이크업 전송 중..."):
                        try:
                            result_img, success = apply_makeup_transfer(st.session_state.original_image, style_image_pil)
                            if success:
                                st.session_state.makeup_image = result_img # 결과 이미지 업데이트 (메이크업 모드 공통 사용)
                                st.session_state.result_caption = f"메이크업 스타일 전송: {selected_style_name}"
                                st.success("✅ 메이크업 스타일 전송 완료!")
                            else:
                                st.error("⚠️ 얼굴 감지 실패 또는 스타일 전송에 문제가 발생했습니다.")
                                # Optionally keep the previous makeup image or reset it
                                # st.session_state.makeup_image = None
                        except Exception as e:
                            st.error(f"메이크업 전송 중 오류 발생: {e}")
                            st.session_state.makeup_image = None

            with col2_tr:
                st.subheader("결과 미리보기 (스타일 전송)")
                # 결과 표시 조건 수정: 현재 모드가 '메이크업'이고, '스타일 전송' 결과가 있을 때
                if st.session_state.makeup_image and st.session_state.result_caption.startswith("메이크업 스타일 전송"):
                    image_comparison(
                        img1=st.session_state.original_image,
                        img2=st.session_state.makeup_image,
                        label1="원본",
                        label2=st.session_state.result_caption,
                        width=700,
                        starting_position=50,
                        show_labels=True
                    )
                    st.divider()
                    save_col1_tr, save_col2_tr = st.columns(2)
                    with save_col1_tr:
                        try:
                            buf = io.BytesIO()
                            save_img_tr = st.session_state.makeup_image.convert('RGB') if st.session_state.makeup_image.mode == 'RGBA' else st.session_state.makeup_image
                            save_img_tr.save(buf, format="PNG")
                            st.download_button("💾 결과 다운로드", buf.getvalue(), f"makeup_transfer_{selected_style_name}.png", "image/png", use_container_width=True)
                        except Exception as e:
                             st.error(f"파일 저장 준비 중 오류: {e}")
                    with save_col2_tr:
                        if st.button("🖼️ 갤러리에 저장", key="save_transfer_gallery", use_container_width=True):
                            st.session_state.gallery.append({'image': st.session_state.makeup_image, 'caption': st.session_state.result_caption})
                            st.success("갤러리에 저장됨!")
                elif apply_transfer_btn: # 버튼 눌렀는데 결과가 없다면
                    st.info("스타일 전송 결과를 기다리는 중이거나 적용에 실패했습니다.")
                    st.image(st.session_state.original_image, caption="원본 이미지", width=400)
                else:
                     st.info("👈 참고할 스타일을 선택하고 '스타일 적용하기' 버튼을 누르세요.")
                     st.image(st.session_state.original_image, caption="원본 이미지", width=400)


    # --- 👕 가상 피팅 모드 ---
    elif st.session_state.app_mode == "가상 피팅":
        # (이전 가상 피팅 코드와 동일)
        # ... (코드 생략) ...
        st.header("👕 AI 가상 피팅")
        col1_vt, col2_vt = st.columns([1, 3]) # 옵션 영역 / 결과 영역

        with col1_vt: # 옵션 설정
            st.subheader("의상 선택 및 옵션")

            if not AVAILABLE_CLOTHING_TYPES:
                st.warning("사용 가능한 의상 샘플이 없습니다. `assets/clothes` 폴더를 확인하거나 URL 다운로드를 확인하세요.")
                selected_clothing_type = None
                clothing_image_pil = None
            else:
                selected_clothing_type = st.selectbox(
                    "의상 종류 선택:",
                     ["의상 선택..."] + AVAILABLE_CLOTHING_TYPES,
                     key="clothing_select"
                )

            clothing_image_pil = None
            if selected_clothing_type and selected_clothing_type != "의상 선택...":
                clothing_image_pil = RESOURCES["clothing"].get(selected_clothing_type)
                if clothing_image_pil:
                    st.image(clothing_image_pil, caption=f"선택된 의상: {selected_clothing_type}", use_container_width=True)
                    st.session_state.tryon_options['selected_clothing'] = selected_clothing_type # 선택된 의상 저장
                else:
                    st.error(f"'{selected_clothing_type}' 의상 이미지를 로드할 수 없습니다.")
                    st.session_state.tryon_options['selected_clothing'] = None
            else:
                 st.session_state.tryon_options['selected_clothing'] = None


            st.divider()
            st.subheader("조정 옵션")
            # 슬라이더 사용 및 범위/기본값 개선
            img_width, img_height = st.session_state.original_image.size
            st.session_state.tryon_options['scale'] = st.slider(
                "크기 조절:", 0.1, 3.0, st.session_state.tryon_options['scale'], 0.05, key="vt_scale"
            )
            # 위치는 이미지 크기를 고려하여 최대/최소 설정
            st.session_state.tryon_options['pos_x'] = st.slider(
                "가로 위치 (X):", -int(img_width*0.5), int(img_width*1.2), st.session_state.tryon_options['pos_x'], 1, key="vt_pos_x"
            )
            st.session_state.tryon_options['pos_y'] = st.slider(
                "세로 위치 (Y):", -int(img_height*0.3), int(img_height*1.2), st.session_state.tryon_options['pos_y'], 1, key="vt_pos_y"
            )

            st.divider()
            st.session_state.tryon_options['color_change'] = st.checkbox("🎨 의상 색상 변경", value=st.session_state.tryon_options['color_change'], key="vt_color_change")
            if st.session_state.tryon_options['color_change']:
                st.session_state.tryon_options['target_color'] = st.color_picker(
                    '변경할 색상', st.session_state.tryon_options['target_color'], key="vt_target_color"
                )

            st.divider()
            apply_tryon_btn = st.button(
                "👕 가상 피팅 적용",
                key="apply_tryon",
                use_container_width=True,
                type="primary",
                disabled=(clothing_image_pil is None)
            )

            if apply_tryon_btn and clothing_image_pil:
                with st.spinner("👔 의상 위치 조정 및 합성 중..."):
                    try:
                        current_clothing_img = clothing_image_pil.copy()
                        caption_suffix = ""

                        # 색상 변경 적용
                        if st.session_state.tryon_options['color_change']:
                            target_color = st.session_state.tryon_options['target_color']
                            current_clothing_img = change_clothing_color(current_clothing_img, target_color)
                            caption_suffix += f" (색상: {target_color})"
                            # st.info(f"의상 색상을 {target_color}(으)로 변경합니다.") # 스피너 중에 표시 잘 안될 수 있음

                        # 가상 피팅 적용
                        position = (st.session_state.tryon_options['pos_x'], st.session_state.tryon_options['pos_y'])
                        scale = st.session_state.tryon_options['scale']
                        result_img = virtual_try_on(st.session_state.original_image, current_clothing_img, position, scale)

                        st.session_state.tryon_image = result_img
                        st.session_state.result_caption = f"가상 피팅: {selected_clothing_type}{caption_suffix}"
                        st.success("✅ 가상 피팅 적용 완료!")

                    except Exception as e:
                        st.error(f"가상 피팅 중 오류 발생: {e}")
                        st.session_state.tryon_image = None

        with col2_vt: # 결과 표시
            st.subheader("결과 미리보기")
            if st.session_state.tryon_image:
                image_comparison(
                    img1=st.session_state.original_image,
                    img2=st.session_state.tryon_image,
                    label1="원본",
                    label2=st.session_state.result_caption,
                    width=700,
                    starting_position=50,
                    show_labels=True
                )
                st.divider()
                save_col1_vt, save_col2_vt = st.columns(2)
                with save_col1_vt:
                    try:
                        buf = io.BytesIO()
                        save_img_vt = st.session_state.tryon_image.convert('RGB') if st.session_state.tryon_image.mode == 'RGBA' else st.session_state.tryon_image
                        save_img_vt.save(buf, format="PNG")
                        st.download_button("💾 결과 다운로드", buf.getvalue(), f"tryon_{selected_clothing_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", "image/png", use_container_width=True)
                    except Exception as e:
                        st.error(f"파일 저장 준비 중 오류: {e}")
                with save_col2_vt:
                    if st.button("🖼️ 갤러리에 저장", key="save_tryon_gallery", use_container_width=True):
                        st.session_state.gallery.append({'image': st.session_state.tryon_image, 'caption': st.session_state.result_caption})
                        st.success("갤러리에 저장됨!")
            elif apply_tryon_btn: # 버튼 눌렀는데 결과가 없다면
                st.info("가상 피팅 결과를 기다리는 중이거나 적용에 실패했습니다.")
                st.image(st.session_state.original_image, caption="원본 이미지", width=400)
            else:
                 st.info("👈 의상을 선택하고 옵션을 조정한 뒤 '가상 피팅 적용' 버튼을 누르세요.")
                 st.image(st.session_state.original_image, caption="원본 이미지", width=400)

# --- END OF FILE app.py ---