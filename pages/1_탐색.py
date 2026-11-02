import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"

# 브라우저 탭 제목과 아이콘 (st.* 명령 중 가장 먼저 실행해야 함)
st.set_page_config(page_title=f"탐색 · {APP_TITLE}", page_icon="🧠", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data201/main/stroke.csv"

COLOR_NO = "#E0B88A"   # 뇌졸중 없음 (따뜻한 베이지)
COLOR_YES = "#C2410C"  # 뇌졸중 있음 (진한 주황)
COLOR_BASE = "#E8944A"
GROUP_COLORS = {"겪지 않음": COLOR_NO, "겪음": COLOR_YES}


# ---------------------------------------------------------------- 데이터
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def style(fig, height=380):
    """그림 공통 모양"""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------- 화면 맨 위
st.title("🔎 탐색")
st.caption(f"{APP_TITLE} · 데이터를 여러 방향에서 살펴봐요")

try:
    df = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})
overall_rate = df["stroke"].mean() * 100

# ---------------------------------------------------------------- 1. 히스토그램
st.header("1. 나이와 평균 혈당의 분포")
col_a, col_b = st.columns(2)

with col_a:
    fig = px.histogram(
        df, x="age", nbins=30, color_discrete_sequence=[COLOR_BASE],
        title="나이 분포", labels={"age": "나이(세)", "count": "사람 수"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(style(fig), width="stretch")

with col_b:
    fig = px.histogram(
        df, x="avg_glucose_level", nbins=40, color_discrete_sequence=[COLOR_BASE],
        title="평균 혈당 분포", labels={"avg_glucose_level": "평균 혈당 (mg/dL)"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(style(fig), width="stretch")

# ---------------------------------------------------------------- 2. 상자그림 + 평균 표
st.header("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
col_c, col_d = st.columns(2)

order = ["겪지 않음", "겪음"]
with col_c:
    fig = px.box(
        df, x="뇌졸중 여부", y="age", color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": order}, color_discrete_map=GROUP_COLORS,
        title="나이 비교", labels={"age": "나이(세)"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(style(fig), width="stretch")

with col_d:
    fig = px.box(
        df, x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": order}, color_discrete_map=GROUP_COLORS,
        title="평균 혈당 비교", labels={"avg_glucose_level": "평균 혈당 (mg/dL)"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(style(fig), width="stretch")

means = (
    df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(order)
    .round(2)
    .rename(columns={"age": "나이 평균(세)", "avg_glucose_level": "평균 혈당의 평균 (mg/dL)"})
)
counts = df["뇌졸중 여부"].value_counts().reindex(order)
means.insert(0, "사람 수", counts.values)
means.index.name = "그룹"
st.subheader("두 그룹의 평균값")
st.dataframe(means.reset_index(), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 3. 고혈압·심장병별 뇌졸중 비율
st.header("3. 고혈압·심장병에 따른 뇌졸중 비율")


def rate_table(column: str) -> pd.DataFrame:
    g = df.groupby(column)["stroke"].agg(["count", "sum", "mean"])
    g = g.reindex([0, 1])
    out = pd.DataFrame(
        {
            "구분": ["없음", "있음"],
            "사람 수": g["count"].astype(int).values,
            "뇌졸중 수": g["sum"].astype(int).values,
            "뇌졸중 비율(%)": (g["mean"] * 100).round(2).values,
        }
    )
    return out


col_e, col_f = st.columns(2)
for col, column, name in ((col_e, "hypertension", "고혈압"), (col_f, "heart_disease", "심장병")):
    t = rate_table(column)
    with col:
        fig = px.bar(
            t, x="구분", y="뇌졸중 비율(%)", text="뇌졸중 비율(%)",
            color="구분", color_discrete_map={"없음": COLOR_NO, "있음": COLOR_YES},
            title=f"{name} 여부별 뇌졸중 비율",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=f"{name}", yaxis_title="뇌졸중 비율(%)")
        fig.update_yaxes(range=[0, max(t["뇌졸중 비율(%)"].max() * 1.25, 1)])
        st.plotly_chart(style(fig), width="stretch")
        st.dataframe(t, hide_index=True, width="stretch")

# ---------------------------------------------------------------- 4. bmi가 빈 사람
st.header("4. 체질량지수(bmi)가 비어 있는 사람")
missing = df[df["bmi"].isna()]
n_missing = len(missing)
rate_missing = missing["stroke"].mean() * 100 if n_missing else float("nan")

compare = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람", "전체"],
        "사람 수": [n_missing, len(df)],
        "뇌졸중 수": [int(missing["stroke"].sum()), int(df["stroke"].sum())],
        "뇌졸중 비율(%)": [round(rate_missing, 2), round(overall_rate, 2)],
    }
)
st.dataframe(compare, hide_index=True, width="stretch")

# ---------------------------------------------------------------- 5. 흡연 상태별 사람 수
st.header("5. 흡연 상태별 사람 수")
smoke = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoke, hide_index=True, width="stretch")
