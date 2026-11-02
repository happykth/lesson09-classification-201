import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"

# 브라우저 탭 제목과 아이콘 (st.* 명령 중 가장 먼저 실행해야 함)
st.set_page_config(page_title=f"분류 모델 · {APP_TITLE}", page_icon="🧠", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data201/main/stroke.csv"

# ---------------------------------------------------------------- 이름·색
FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
KO = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LOGI_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델"

COLOR_NEG_POINT = "#5B7C99"   # 실제 뇌졸중 아님 (점)
COLOR_POS_POINT = "#C2410C"   # 실제 뇌졸중 (점)
COLOR_NEG_AREA = "#FFF1DE"    # 트리가 '아님'이라 답하는 칸 (옅은 색)
COLOR_POS_AREA = "#F9C9A8"    # 트리가 '뇌졸중'이라 답하는 칸 (옅은 색)
COLOR_NEG_LEAF = "#FDE8CF"
COLOR_POS_LEAF = "#F28C5B"

# ---------------------------------------------------------------- 스타일
st.markdown(
    """
    <style>
    .card {
        background: #FFF4E6;
        border: 1px solid #F3D5B0;
        border-radius: 16px;
        padding: 18px 14px 14px 14px;
        text-align: center;
    }
    .card-name { color: #7C4A1E; font-size: 0.95rem; font-weight: 700; min-height: 3.2em; }
    .card-label { color: #7C4A1E; font-size: 0.85rem; margin-top: 4px; }
    .card-value { color: #C2410C; font-size: 2.6rem; font-weight: 800; line-height: 1.15; }
    .card-small {
        display: flex; justify-content: space-between; gap: 8px;
        margin-top: 8px; padding-top: 8px; border-top: 1px dashed #F3D5B0;
        color: #7C4A1E; font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- 데이터
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def fmt(v) -> str:
    return f"{v:g}"


def pct(v) -> str:
    return f"{v * 100:.2f}%"


# ---------------------------------------------------------------- 화면 맨 위
st.title("🤖 분류 모델")
st.caption(f"{APP_TITLE} · 뇌졸중(stroke = 1)을 양성으로 두고, 두 모델이 얼마나 잘 맞히는지 비교해요")

try:
    raw = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# ---------------------------------------------------------------- 1. 속성 고르기
st.header("1. 입력으로 사용할 속성 고르기")
chosen = st.multiselect(
    "입력으로 사용할 속성 (두 개 이상)",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda k: KO[k],
)
selected = [k for k in FEATURES if k in chosen]  # 항상 같은 순서로 정리

if len(selected) < 2:
    st.info("속성을 두 개 이상 골라 주세요. 그래야 두 모델을 만들고 그림을 그릴 수 있어요.")
    st.stop()

# ---------------------------------------------------------------- 2. 나누기 / 채우기 / 크기 맞추기
data = raw.sort_values("id").reset_index(drop=True)          # 번호 순 정렬
is_test = (np.arange(len(data)) % 10) < 3                    # 열 명씩 묶어 앞 세 명 = 테스트용

X_train = data.loc[~is_test, selected].copy()
X_test = data.loc[is_test, selected].copy()
y_train = data.loc[~is_test, "stroke"].astype(int).to_numpy()
y_test = data.loc[is_test, "stroke"].astype(int).to_numpy()

bmi_note = ""
if "bmi" in selected:
    n_na_train = int(X_train["bmi"].isna().sum())
    n_na_test = int(X_test["bmi"].isna().sum())
    bmi_median = float(X_train["bmi"].median())               # 훈련용의 중앙값
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    bmi_note = (
        f" 체질량지수는 비어 있는 사람(훈련용 {n_na_train}명, 테스트용 {n_na_test}명)을 "
        f"훈련용의 중앙값 {bmi_median:.1f}로 채웠어요."
    )

st.caption(
    f"번호 순으로 정렬해 열 명씩 묶고 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"전체 {len(data):,}명 → 훈련용 {len(X_train):,}명, 테스트용 {len(X_test):,}명."
    f"{bmi_note} 로지스틱 회귀는 훈련용으로만 구한 평균과 흩어진 정도로 크기를 맞춘 뒤 학습해요."
)

# ---------------------------------------------------------------- 3. 모델 만들기
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logit.fit(X_train, y_train)

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

base = DummyClassifier(strategy="most_frequent")
base.fit(X_train, y_train)

# ---------------------------------------------------------------- 4. 정확도 카드
st.header("2. 정확도 비교")


def acc_card(name: str, model) -> str:
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    return (
        f'<div class="card">'
        f'<div class="card-name">{name}</div>'
        f'<div class="card-label">테스트 데이터 정확도</div>'
        f'<div class="card-value">{pct(test_acc)}</div>'
        f'<div class="card-small"><span>훈련 데이터 {pct(train_acc)}</span>'
        f"<span>테스트 데이터 {pct(test_acc)}</span></div>"
        f"</div>"
    )


c1, c2, c3 = st.columns(3)
c1.markdown(acc_card(LOGI_NAME, logit), unsafe_allow_html=True)
c2.markdown(acc_card(TREE_NAME, tree), unsafe_allow_html=True)
c3.markdown(acc_card(BASE_NAME, base), unsafe_allow_html=True)

# ---------------------------------------------------------------- 5. 산점도
st.header("3. 두 속성으로 보는 모델의 판단")

col_x, col_y = st.columns(2)
with col_x:
    x_key = st.selectbox("가로축", selected, index=0, format_func=lambda k: KO[k])
with col_y:
    y_options = [k for k in selected if k != x_key]
    y_key = st.selectbox("세로축", y_options, index=0, format_func=lambda k: KO[k])

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둔다
med_test = X_test.median()
others = [k for k in selected if k not in (x_key, y_key)]


def make_frame(xv, yv) -> pd.DataFrame:
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    frame = pd.DataFrame({c: np.full(len(xv), float(med_test[c])) for c in selected})
    frame[x_key] = xv
    frame[y_key] = yv
    return frame[selected]


def axis_range(key: str):
    if key in BINARY:
        return -0.5, 1.5
    lo, hi = float(X_test[key].min()), float(X_test[key].max())
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    return lo - pad, hi + pad


x0, x1 = axis_range(x_key)
y0, y1 = axis_range(y_key)

# --- 로지스틱 회귀의 0.5 경계선: a*x + b*y + c = 0 (확률이 0.5인 곳)
probe = logit.decision_function(make_frame([0, 1, 0], [0, 0, 1]))
c_ = float(probe[0])
a_ = float(probe[1] - probe[0])
b_ = float(probe[2] - probe[0])

cands = []
eps = 1e-9
if abs(b_) > eps:
    for xv in (x0, x1):
        yv = -(a_ * xv + c_) / b_
        if y0 - eps <= yv <= y1 + eps:
            cands.append((xv, yv))
if abs(a_) > eps:
    for yv in (y0, y1):
        xv = -(b_ * yv + c_) / a_
        if x0 - eps <= xv <= x1 + eps:
            cands.append((xv, yv))

line_pts = None
if len(cands) >= 2:
    best = -1.0
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            d = (cands[i][0] - cands[j][0]) ** 2 + (cands[i][1] - cands[j][1]) ** 2
            if d > best:
                best, line_pts = d, (cands[i], cands[j])
    if best <= 1e-18:
        line_pts = None

# --- 의사결정트리가 그림을 나눈 칸
xs = np.linspace(x0, x1, 220)
ys = np.linspace(y0, y1, 220)
XX, YY = np.meshgrid(xs, ys)
region = tree.predict(make_frame(XX.ravel(), YY.ravel())).reshape(XX.shape)
present = set(np.unique(region).tolist())

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=xs, y=ys, z=region, zmin=0, zmax=1,
        colorscale=[[0, COLOR_NEG_AREA], [1, COLOR_POS_AREA]],
        showscale=False, hoverinfo="skip", showlegend=False,
    )
)
# 옅은 칸의 범례(작은 네모)
for cls, color, label in (
    (0, COLOR_NEG_AREA, "트리가 '뇌졸중 아님'이라 답하는 칸"),
    (1, COLOR_POS_AREA, "트리가 '뇌졸중'이라 답하는 칸"),
):
    if cls in present:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", name=label,
                marker=dict(symbol="square", size=14, color=color, line=dict(color="#D9B58C", width=1)),
            )
        )

# 테스트 데이터 점 (0/1 같은 축은 겹치지 않게 보기용으로만 살짝 흩어 놓음)
rng = np.random.default_rng(0)


def jitter(values: np.ndarray, key: str) -> np.ndarray:
    return values + rng.uniform(-0.12, 0.12, len(values)) if key in BINARY else values


for cls, color, label, size in (
    (0, COLOR_NEG_POINT, "실제 뇌졸중 아님 (0)", 6),
    (1, COLOR_POS_POINT, "실제 뇌졸중 (1)", 8),
):
    m = y_test == cls
    xv = X_test.loc[:, x_key].to_numpy()[m]
    yv = X_test.loc[:, y_key].to_numpy()[m]
    fig.add_trace(
        go.Scatter(
            x=jitter(xv, x_key), y=jitter(yv, y_key), mode="markers", name=label,
            customdata=np.column_stack([xv, yv]),
            hovertemplate=f"{KO[x_key]}: %{{customdata[0]:g}}<br>{KO[y_key]}: %{{customdata[1]:g}}<extra>{label}</extra>",
            marker=dict(size=size, color=color, opacity=0.65 if cls == 0 else 0.95,
                        line=dict(color="white", width=0.5)),
        )
    )

if line_pts is not None:
    fig.add_trace(
        go.Scatter(
            x=[line_pts[0][0], line_pts[1][0]], y=[line_pts[0][1], line_pts[1][1]],
            mode="lines", name="로지스틱 회귀가 0.5로 가르는 선",
            line=dict(color="#1F2937", width=3, dash="dash"),
        )
    )

fig.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title=KO[x_key], range=[x0, x1], showgrid=False, zeroline=False, automargin=True),
    yaxis=dict(title=KO[y_key], range=[y0, y1], showgrid=False, zeroline=False, automargin=True),
    legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
)
for key, setter in ((x_key, fig.update_xaxes), (y_key, fig.update_yaxes)):
    if key in BINARY:
        setter(tickmode="array", tickvals=[0, 1], ticktext=["없음", "있음"])
st.plotly_chart(fig, width="stretch")

# --- 그림 아래 설명 글
if others:
    parts = []
    for k in others:
        v = float(med_test[k])
        parts.append(f"{KO[k]} {int(v)}({'있음' if v >= 0.5 else '없음'})" if k in BINARY else f"{KO[k]} {fmt(round(v, 2))}")
    st.write(
        "그림에 나오지 않는 속성은 **테스트 데이터의 중앙값**에 세워 두고 계산했어요: "
        + ", ".join(parts) + ". (옅은 색 칸과 점선 모두 이 값을 기준으로 해요.)"
    )
else:
    st.write("고른 속성이 두 개뿐이라 따로 중앙값에 세워 둘 속성은 없어요.")

if line_pts is None:
    center = (a_ * (x0 + x1) / 2 + b_ * (y0 + y1) / 2 + c_) > 0
    side = "'뇌졸중'" if center else "'뇌졸중 아님'"
    st.write(
        f"로지스틱 회귀가 확률 0.5로 가르는 선은 **이 그림 밖**에 있어요. "
        f"그림 안은 모두 {side} 쪽(확률이 0.5 {'이상' if center else '미만'})이에요."
    )
else:
    st.write("점선은 로지스틱 회귀가 뇌졸중일 확률을 0.5로 보는 자리예요. 선의 한쪽은 '뇌졸중', 다른 쪽은 '아님'으로 답해요.")

if len(present) == 1:
    only = "'뇌졸중'" if 1 in present else "'뇌졸중 아님'"
    st.write(f"이 그림 안에서는 의사결정트리가 모든 칸을 {only}이라고 답해서 옅은 색이 한 가지로만 칠해졌어요.")

if x_key in BINARY or y_key in BINARY:
    st.caption("고혈압·심장병은 값이 0과 1뿐이라, 점이 겹치지 않게 보기용으로만 살짝 흩어 놓았어요. 계산에는 원래 값을 썼어요.")

# ---------------------------------------------------------------- 6. 가지 그림
st.header("4. 의사결정트리가 던진 질문")

t = tree.tree_
indicator = tree.decision_path(X_train)                       # (사람, 마디) 표
node_total = np.asarray(indicator.sum(axis=0)).ravel().astype(int)
node_pos = np.asarray(indicator.T.dot(y_train)).ravel().astype(int)

LEAF = -1
lines = [
    "digraph T {",
    '  graph [rankdir=LR, nodesep=0.25, ranksep=0.9];',
    '  node [shape=box, style="rounded,filled", fontsize=16, margin="0.16,0.08", color="#C9A27A"];',
    '  edge [fontsize=15, color="#8A6A4A"];',
]
n_leaf = 0
n_leaf_neg = 0
used_idx = set()

for i in range(t.node_count):
    total, pos = int(node_total[i]), int(node_pos[i])
    rate = pos / total * 100 if total else 0.0
    body = f"훈련용 {total:,}명\\n뇌졸중 {pos:,}명 ({rate:.1f}%)"
    if t.children_left[i] == LEAF:
        n_leaf += 1
        answer_pos = pos > total - pos          # 많은 쪽으로 답함 (같으면 '아님')
        if not answer_pos:
            n_leaf_neg += 1
        answer = "뇌졸중" if answer_pos else "뇌졸중 아님"
        fill = COLOR_POS_LEAF if answer_pos else COLOR_NEG_LEAF
        lines.append(f'  n{i} [label="{body}\\n답: {answer}", fillcolor="{fill}"];')
    else:
        feat_idx = int(t.feature[i])
        used_idx.add(feat_idx)
        key = selected[feat_idx]
        if key in BINARY:
            question = f"{KO[key]}이 있나요?"
        else:
            question = f"{KO[key]} ≤ {t.threshold[i]:.1f} ?"
        lines.append(f'  n{i} [label="{question}\\n{body}", fillcolor="#FFFFFF"];')

for i in range(t.node_count):
    left, right = int(t.children_left[i]), int(t.children_right[i])
    if left == LEAF:
        continue
    key = selected[int(t.feature[i])]
    if key in BINARY:       # 값이 1(있음)이면 오른쪽
        yes, no = right, left
    else:                   # 기준 이하이면 왼쪽
        yes, no = left, right
    lines.append(f'  n{i} -> n{yes} [label="예"];')
    lines.append(f'  n{i} -> n{no} [label="아니요"];')
lines.append("}")

st.graphviz_chart("\n".join(lines), width="stretch")

leaf_summary = f"답을 내는 마디는 모두 **{n_leaf}칸**이고, 그중 **{n_leaf_neg}칸**이 '뇌졸중 아님'이라고 답해요."
if n_leaf > n_leaf_neg:
    leaf_summary += f" 나머지 {n_leaf - n_leaf_neg}칸은 '뇌졸중'이라고 답해요."
st.write(leaf_summary)
if used_idx:
    used_names = [KO[selected[i]] for i in sorted(used_idx)]
    st.write(f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것은 **{', '.join(used_names)}**이에요.")
else:
    st.write(f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것은 **없어요**. 한 번도 나누지 않고 바로 답을 냈어요.")
