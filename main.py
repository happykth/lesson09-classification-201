import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"

# 브라우저 탭 제목과 아이콘 (st.* 명령 중 가장 먼저 실행해야 함)
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data201/main/stroke.csv"
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "데이터_소개.json"

# ---------------------------------------------------------------- 스타일
st.markdown(
    """
    <style>
    .card {
        background: #FFF4E6;
        border: 1px solid #F3D5B0;
        border-radius: 16px;
        padding: 22px 12px;
        text-align: center;
    }
    .card-label { color: #7C4A1E; font-size: 1.0rem; margin-bottom: 6px; }
    .card-value { color: #C2410C; font-size: 2.6rem; font-weight: 800; line-height: 1.1; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- 데이터
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def fmt(x) -> str:
    """숫자를 보기 좋게 (3.0 -> 3, 28.5 -> 28.5)"""
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def value_kind(s: pd.Series) -> str:
    """열에 들어 있는 값의 종류를 한 줄로 설명"""
    values = s.dropna()
    uniques = sorted(values.unique(), key=lambda v: str(v))
    if values.dtype == object or len(uniques) <= 5:
        return ", ".join(fmt(v) for v in sorted(uniques, key=lambda v: (str(type(v)), v)))
    return f"숫자 ({fmt(values.min())} ~ {fmt(values.max())})"


# ---------------------------------------------------------------- 저장 / 불러오기
def load_notes() -> dict:
    try:
        with open(SAVE_FILE, encoding="utf-8") as f:
            notes = json.load(f)
        return {
            "meanings": dict(notes.get("meanings", {})),
            "source": str(notes.get("source", "")),
        }
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"meanings": {}, "source": ""}


def save_notes(meanings: dict, source: str) -> None:
    SAVE_DIR.mkdir(exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({"meanings": meanings, "source": source}, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- 화면 맨 위
st.title(f"🧠 {APP_TITLE}")
st.caption("첫 화면 · 이 데이터는 무엇일까요?")

try:
    df = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# ---------------------------------------------------------------- 큰 숫자 카드 4개
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100


def card(label: str, value: str) -> str:
    return f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div></div>'


c1, c2, c3, c4 = st.columns(4)
c1.markdown(card("전체 사람 수", f"{total:,}명"), unsafe_allow_html=True)
c2.markdown(card("열 개수", f"{n_cols}개"), unsafe_allow_html=True)
c3.markdown(card("stroke가 1인 사람", f"{n_stroke:,}명"), unsafe_allow_html=True)
c4.markdown(card("stroke 1의 비율", f"{ratio:.2f}%"), unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------- 열 설명 표 + 출처 (저장 폼)
notes = load_notes()

table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [notes["meanings"].get(c, "") for c in df.columns],
        "값의 종류": [value_kind(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

with st.form("notes_form"):
    st.subheader("📋 열 설명표")
    st.caption("‘우리말 뜻’ 칸을 두 번 눌러 교재를 보고 직접 적어 보세요.")
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름"),
            "우리말 뜻": st.column_config.TextColumn("우리말 뜻 (직접 입력)"),
            "값의 종류": st.column_config.TextColumn("값의 종류"),
            "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
        },
        key="meaning_editor",
    )

    # ------------------------------------------------------------ 처음 다섯 줄
    st.subheader("👀 데이터 처음 다섯 줄")
    st.dataframe(df.head(5), hide_index=True, width="stretch")

    # ------------------------------------------------------------ 출처
    st.subheader("📚 데이터 출처")
    source_text = st.text_area(
        "출처",
        value=notes["source"],
        height=120,
        placeholder="교재에 적힌 데이터 출처를 여기에 적어 주세요.",
        label_visibility="collapsed",
    )

    saved = st.form_submit_button("💾 저장하기", type="primary")

if saved:
    meanings = {
        str(name): ("" if pd.isna(text) else str(text))
        for name, text in zip(edited["열 이름"], edited["우리말 뜻"])
    }
    try:
        save_notes(meanings, source_text)
        st.success("내_기록 폴더에 저장했어요. 새로고침해도 다시 불러옵니다.")
    except OSError as e:
        st.error(f"저장하지 못했어요: {e}")
