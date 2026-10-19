import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 설정 및 제목 (브라우저 탭 제목 & 아이콘)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")

# 2. 데이터 불러오기 (캐싱 활용)
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df_raw = load_data()
total_count = len(df_raw)

# 3. 데이터 전처리 및 속성 생성
df = df_raw.copy()

# 수치형 컬럼 변환
numeric_cols = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 필터링 조건: 네 속성 결측치 제외 & 관객/스크린 수가 0 이하인 행 제외
valid_mask = (
    df['first_scrn'].notna() & (df['first_scrn'] > 0) &
    df['total_audi'].notna() & (df['total_audi'] > 0) &
    df['days_in_top10'].notna() &
    df['first_week_audi'].notna() & (df['first_week_audi'] > 0)
)
df_clean = df[valid_mask].copy()

# 파생 속성 생성
# - 롱런 지수 = 누적 관객 / 첫 주 관객 (20 초과 시 20으로 클리핑)
df_clean['long_run_index'] = (df_clean['total_audi'] / df_clean['first_week_audi']).clip(upper=20)
# - 스크린 수 (상용로그)
df_clean['log_first_scrn'] = np.log10(df_clean['first_scrn'])
# - 누적 관객 (상용로그)
df_clean['log_total_audi'] = np.log10(df_clean['total_audi'])

filtered_count = len(df_clean)

# 전체 편수와 묶은 편수를 한 줄로 표시
st.markdown(f"**전체 영화 편수:** {total_count}편 | **분석(묶은) 대상 영화 편수:** {filtered_count}편")
st.divider()

# 4. 군집화 속성 선택 UI
FEATURE_DICT = {
    "스크린 수 (상용로그)": "log_first_scrn",
    "누적 관객 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index"
}

selected_feature_labels = st.multiselect(
    "군집화(묶기)에 사용할 속성을 선택하세요 (2개 이상 선택 가능, 기본은 4개 모두 선택):",
    options=list(FEATURE_DICT.keys()),
    default=list(FEATURE_DICT.keys())
)

if len(selected_feature_labels) < 2:
    st.warning("⚠️ 군집화를 수행하려면 최소 2개 이상의 속성을 선택해 주세요.")
    st.stop()

# 5. 선택된 속성 표준화 및 K-Means 군집화 (고정된 난수 시드)
selected_cols = [FEATURE_DICT[lbl] for lbl in selected_feature_labels]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[selected_cols])

kmeans = KMeans(n_clusters=3, random_state=42)
raw_clusters = kmeans.fit_predict(X_scaled)
df_clean['raw_cluster'] = raw_clusters

# 6. 묶음 라벨 재할당 (누적 관객 평균이 큰 묶음부터 ㉮, ㉯, ㉰)
cluster_means = df_clean.groupby('raw_cluster')['total_audi'].mean().sort_values(ascending=False)
cluster_mapping = {raw_id: label for raw_id, label in zip(cluster_means.index, ['㉮', '㉯', '㉰'])}
df_clean['cluster'] = df_clean['raw_cluster'].map(cluster_mapping)

# 7. 2차원 산점도 (Plotly)
st.subheader("📌 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_2d_label = st.selectbox("가로축 (X축) 속성", options=selected_feature_labels, index=0, key="2d_x")
with col2:
    y_2d_label = st.selectbox("세로축 (Y축) 속성", options=selected_feature_labels, index=min(1, len(selected_feature_labels)-1), key="2d_y")

x_2d_col = FEATURE_DICT[x_2d_label]
y_2d_col = FEATURE_DICT[y_2d_label]

fig_2d = px.scatter(
    df_clean,
    x=x_2d_col,
    y=y_2d_col,
    color='cluster',
    hover_name='movieNm',
    category_orders={'cluster': ['㉮', '㉯', '㉰']},
    labels={x_2d_col: x_2d_label, y_2d_col: y_2d_label, 'cluster': '묶음'},
    title=f"2차원 산점도: {x_2d_label} vs {y_2d_label}"
)
st.plotly_chart(fig_2d, use_container_width=True)

# 8. 3차원 산점도 (Plotly)
st.subheader("📌 3차원 산점도")
if len(selected_feature_labels) < 3:
    st.info("💡 3차원 산점도를 그리려면 선택된 속성이 3개 이상이어야 합니다.")
else:
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        x_3d_label = st.selectbox("X축 속성", options=selected_feature_labels, index=0, key="3d_x")
    with col_y:
        y_3d_label = st.selectbox("Y축 속성", options=selected_feature_labels, index=1, key="3d_y")
    with col_z:
        z_3d_label = st.selectbox("Z축 속성", options=selected_feature_labels, index=2, key="3d_z")

    x_3d_col = FEATURE_DICT[x_3d_label]
    y_3d_col = FEATURE_DICT[y_3d_label]
    z_3d_col = FEATURE_DICT[z_3d_label]

    fig_3d = px.scatter_3d(
        df_clean,
        x=x_3d_col,
        y=y_3d_col,
        z=z_3d_col,
        color='cluster',
        hover_name='movieNm',
        category_orders={'cluster': ['㉮', '㉯', '㉰']},
        labels={x_3d_col: x_3d_label, y_3d_col: y_3d_label, z_3d_col: z_3d_label, 'cluster': '묶음'},
        title=f"3차원 산점도: {x_3d_label} vs {y_3d_label} vs {z_3d_label}"
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# 9. 묶음별 통계 표 (원래 단위 기준)
st.subheader("📊 묶음별 편수 및 속성 평균 (원래 단위)")

stats_data = []
for cluster_label in ['㉮', '㉯', '㉰']:
    cdf = df_clean[df_clean['cluster'] == cluster_label]
    stats_data.append({
        "묶음": cluster_label,
        "편수": len(cdf),
        "스크린 수 평균": round(cdf['first_scrn'].mean(), 1),
        "누적 관객 평균": round(cdf['total_audi'].mean(), 1),
        "10위권 일수 평균": round(cdf['days_in_top10'].mean(), 1),
        "롱런 지수 평균": round(cdf['long_run_index'].mean(), 2)
    })

stats_df = pd.DataFrame(stats_data)
st.dataframe(stats_df, use_container_width=True)

# 10. 묶음별 누적 관객 TOP 5 영화
st.subheader("🏆 묶음별 누적 관객 TOP 5 영화")

col_a, col_b, col_c = st.columns(3)
cluster_cols = {'㉮': col_a, '㉯': col_b, '㉰': col_c}

for cluster_label in ['㉮', '㉯', '㉰']:
    with cluster_cols[cluster_label]:
        st.markdown(f"#### 묶음 {cluster_label}")
        top5_df = (
            df_clean[df_clean['cluster'] == cluster_label]
            .sort_values(by='total_audi', ascending=False)
            .head(5)
        )
        for idx, row in enumerate(top5_df.itertuples(), start=1):
            st.markdown(f"{idx}. **{row.movieNm}** ({row.total_audi:,.0f}명)")
