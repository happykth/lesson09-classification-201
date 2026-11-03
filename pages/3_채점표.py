import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"

# 브라우저 탭 제목과 아이콘 (st.* 명령 중 가장 먼저 실행해야 함)
st.set_page_config(page_title=f"채점표 · {APP_TITLE}", page_icon="🧠", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data201/main/stroke.csv"

# ---------------------------------------------------------------- 이름
FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
KO = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LOGI_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "훈련용에서 많은 쪽으로만 답하는 모델"

CANNOT = "계산할 수 없음"
MAX_ROWS = 20

# ---------------------------------------------------------------- 스타일
st.markdown(
    """
    <style>
    table.cm { border-collapse: separate; border-spacing: 4px; width: 100%; table-layout: fixed; }
    table.cm th { background: #F6E7D2; color: #7C4A1E; font-weight: 700; padding: 8px 4px;
                  text-align: center; border-radius: 8px; font-size: 0.9rem; }
    table.cm td { text-align: center; padding: 10px 4px; border-radius: 10px; color: #3B2A1A; }
    table.cm td.good { background: #FFF4E6; border: 1px solid #F3D5B0; }
    table.cm td.bad  { background: #FDE0D0; border: 1px solid #F0B79A; }
    table.cm td.sum  { background: #F4F1EC; color: #5A4A3A; }
    table.cm .abbr { color: #C2410C; font-weight: 800; font-size: 1.05rem; }
    table.cm .hint { color: #8A6A4A; font-size: 0.75rem; }
    table.cm .num  { font-size: 1.15rem; font-weight: 800; line-height: 1.2; white-space: nowrap; }
    table.cm td.sum .num { font-size: 1.15rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- 데이터
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def pct(v) -> str:
    return CANNOT if v is None else f"{v * 100:.2f}%"


# ---------------------------------------------------------------- 화면 맨 위
st.title("📝 채점표")
st.caption(f"{APP_TITLE} · 테스트 데이터로 두 모델을 채점해 봐요 (뇌졸중을 양성으로 둬요)")

try:
    raw = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# ---------------------------------------------------------------- 입력 속성 (분류 모델 페이지와 같은 방법)
chosen = st.multiselect(
    "입력으로 사용할 속성 (두 개 이상)",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda k: KO[k],
)
selected = [k for k in FEATURES if k in chosen]

if len(selected) < 2:
    st.info("속성을 두 개 이상 골라 주세요. 그래야 두 모델을 만들고 채점할 수 있어요.")
    st.stop()

# ---------------------------------------------------------------- 나누기 / 채우기 (분류 모델 페이지와 같은 방법)
data = raw.sort_values("id").reset_index(drop=True)          # 번호 순 정렬
is_test = (np.arange(len(data)) % 10) < 3                    # 열 명씩 묶어 앞 세 명 = 테스트용

X_train = data.loc[~is_test, selected].copy()
X_test = data.loc[is_test, selected].copy()
y_train = data.loc[~is_test, "stroke"].astype(int).to_numpy()
y_test = data.loc[is_test, "stroke"].astype(int).to_numpy()

if "bmi" in selected:
    bmi_median = float(X_train["bmi"].median())               # 훈련용의 중앙값
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# ---------------------------------------------------------------- 모델 학습 (분류 모델 페이지와 같은 방법)
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logit.fit(X_train, y_train)

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

base = DummyClassifier(strategy="most_frequent")
base.fit(X_train, y_train)

models = [(LOGI_NAME, logit), (TREE_NAME, tree), (BASE_NAME, base)]
preds = {name: np.asarray(m.predict(X_test)).astype(int) for name, m in models}


def confusion(pred: np.ndarray) -> dict:
    """뇌졸중(1)을 양성으로 둔 네 칸"""
    return {
        "TP": int(((y_test == 1) & (pred == 1)).sum()),
        "FN": int(((y_test == 1) & (pred == 0)).sum()),
        "FP": int(((y_test == 0) & (pred == 1)).sum()),
        "TN": int(((y_test == 0) & (pred == 0)).sum()),
    }


cms = {name: confusion(p) for name, p in preds.items()}

n_test = len(y_test)
n_test_pos = int((y_test == 1).sum())

# ---------------------------------------------------------------- 1. 혼동행렬
st.header("1. 혼동행렬")
st.write(f"테스트용 **{n_test:,}명** 가운데 실제 뇌졸중인 사람은 **{n_test_pos:,}명**이에요.")
st.caption("행은 실제, 열은 예측이에요. 칸 안의 글자는 TP(뇌졸중을 맞힘)·FN(뇌졸중을 놓침)·FP(헛짚음)·TN(아니라고 맞힘)이에요.")


def cm_html(c: dict) -> str:
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]

    def cell(abbr: str, hint: str, n: int, good: bool) -> str:
        return (
            f'<td class="{"good" if good else "bad"}">'
            f'<div class="abbr">{abbr}</div><div class="hint">{hint}</div>'
            f'<div class="num">{n:,}명</div></td>'
        )

    def total(n: int) -> str:
        return f'<td class="sum"><div class="num">{n:,}명</div></td>'

    return (
        '<table class="cm">'
        '<tr><th colspan="2" rowspan="2"></th><th colspan="2">예측</th><th rowspan="2">합계</th></tr>'
        "<tr><th>뇌졸중</th><th>아님</th></tr>"
        '<tr><th rowspan="2">실제</th><th>뇌졸중</th>'
        + cell("TP", "뇌졸중을 맞힘", tp, True)
        + cell("FN", "뇌졸중을 놓침", fn, False)
        + total(tp + fn)
        + "</tr>"
        "<tr><th>아님</th>"
        + cell("FP", "아닌데 뇌졸중이라 예측", fp, False)
        + cell("TN", "아니라고 맞힘", tn, True)
        + total(fp + tn)
        + "</tr>"
        '<tr><th colspan="2">합계</th>'
        + total(tp + fp)
        + total(fn + tn)
        + total(tp + fn + fp + tn)
        + "</tr></table>"
    )


col_l, col_r = st.columns(2)
for col, name in ((col_l, LOGI_NAME), (col_r, TREE_NAME)):
    with col:
        st.subheader(name)
        st.markdown(cm_html(cms[name]), unsafe_allow_html=True)

# ---------------------------------------------------------------- 2. 네 가지 지표
st.header("2. 네 가지 지표")


def scores(c: dict):
    """정확도·재현율·정밀도·F1 (계산할 수 없으면 None)"""
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None          # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None       # 분모: 뇌졸중이라 예측한 사람 수
    zero_pr = False
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1, zero_pr = 0.0, True      # 공식의 분모가 0이지만 맞힌 환자가 없으므로 0으로 적음
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1, zero_pr


rows = []
zero_pr_models = []
for name, _ in models:
    accuracy, recall, precision, f1, zero_pr = scores(cms[name])
    if zero_pr:
        zero_pr_models.append(name)
    rows.append(
        {"모델": name, "정확도": pct(accuracy), "재현율": pct(recall), "정밀도": pct(precision), "F1": pct(f1)}
    )

st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(
    "정확도 = (TP+TN) ÷ 전체 사람 수 · 재현율 = TP ÷ 실제 뇌졸중인 사람 수(TP+FN) · "
    "정밀도 = TP ÷ 뇌졸중이라 예측한 사람 수(TP+FP) · F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)
if zero_pr_models:
    st.caption(
        f"{', '.join(zero_pr_models)}은(는) 정밀도와 재현율이 모두 0이라 공식의 분모가 0이 되지만, "
        "맞힌 뇌졸중 환자가 한 명도 없으므로 F1을 0으로 적었어요."
    )

# ---------------------------------------------------------------- 3. 놓친 환자 / 헛짚은 사람
st.header("3. 놓친 환자와 헛짚은 사람")
model_name = st.radio(
    "어느 모델의 목록을 볼까요?",
    [LOGI_NAME, TREE_NAME],
    horizontal=True,
)
pred = preds[model_name]
cm = cms[model_name]

test_rows = data.loc[is_test, ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]].reset_index(drop=True)


def people(mask: np.ndarray) -> pd.DataFrame:
    sub = test_rows[mask].sort_values(["age", "id"], ascending=[False, True])   # 나이 많은 순
    return pd.DataFrame(
        {
            "번호": sub["id"].astype(int).to_numpy(),
            "나이": sub["age"].to_numpy(),
            "평균 혈당": sub["avg_glucose_level"].to_numpy(),
            "고혈압 여부": sub["hypertension"].map({0: "없음", 1: "있음"}).to_numpy(),
            "심장병 여부": sub["heart_disease"].map({0: "없음", 1: "있음"}).to_numpy(),
        }
    )


missed = people((y_test == 1) & (pred == 0))      # 놓친 환자 (FN)
wrong = people((y_test == 0) & (pred == 1))       # 헛짚은 사람 (FP)


def show_list(col, title: str, desc: str, table: pd.DataFrame) -> None:
    with col:
        st.subheader(f"{title} · 전체 {len(table):,}명")
        note = f"{desc}"
        if len(table) > MAX_ROWS:
            note += f" · 나이 많은 순으로 앞 {MAX_ROWS}명만 보여 줘요"
        else:
            note += " · 나이 많은 순"
        st.caption(note)
        if table.empty:
            st.info("해당하는 사람이 없어요.")
            return
        shown = table.head(MAX_ROWS)
        st.dataframe(
            shown,
            hide_index=True,
            width="stretch",
            height=(len(shown) + 1) * 35 + 3,
            column_config={
                "번호": st.column_config.NumberColumn("번호", format="%d"),
                "나이": st.column_config.NumberColumn("나이", format="%g"),
                "평균 혈당": st.column_config.NumberColumn("평균 혈당", format="%.2f"),
            },
        )


list_l, list_r = st.columns(2)
show_list(list_l, "놓친 환자", "실제 뇌졸중인데 아니라고 예측한 사람", missed)
show_list(list_r, "헛짚은 사람", "실제로는 아닌데 뇌졸중이라고 예측한 사람", wrong)

same = len(missed) == cm["FN"] and len(wrong) == cm["FP"]
st.write(
    f"왼쪽 목록 {len(missed):,}명은 혼동행렬의 FN {cm['FN']:,}명, "
    f"오른쪽 목록 {len(wrong):,}명은 FP {cm['FP']:,}명과 "
    + ("같아요." if same else "달라요. 확인이 필요해요.")
)
