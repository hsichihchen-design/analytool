import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import numpy as np
import random
from datetime import datetime, timedelta
import google.generativeai as genai
import json

# ==========================================
# 0. 全域設定與 Helper Functions
# ==========================================
st.set_page_config(page_title="作圖小工具 V140", layout="wide", page_icon="✨")

# 安全獲取 Index 的小工具
def safe_index(val, options):
    try: return options.index(val)
    except: return 0

# ==========================================
# 1. 初始化 Session State (索引驅動核心)
# ==========================================
# 我們只追蹤 IDX (整數索引)，這是最穩定的方式
state_keys = [
    'chart_type_idx', 'x_col_idx', 'y_col_idx', 'y_col_2_idx', 
    'color_col_idx', 'agg_func_idx', 'sort_order_idx',
    'treemap_path', 'gemini_api_key', 'ai_insights', 'last_analyzed_file'
]

for key in state_keys:
    if key not in st.session_state:
        if 'idx' in key: st.session_state[key] = 0
        elif 'path' in key: st.session_state[key] = []
        else: st.session_state[key] = None

# ==========================================
# 2. CSS 樣式 (V140 Layout)
# ==========================================
def inject_custom_css(font_family):
    google_font_import = ""
    font_css_rule = font_family
    if font_family == "Noto Sans TC (推薦)":
        google_font_import = "@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap');"
        font_css_rule = "'Noto Sans TC', sans-serif"
    elif "華康" in font_family:
        font_css_rule = f"'{font_family}', 'Microsoft JhengHei', sans-serif"

    st.markdown(f"""
    <style>
        {google_font_import}
        html, body, [class*="css"] {{ font-family: {font_css_rule} !important; }}
        
        /* 確保頂部留白足夠，標題不被遮擋 */
        .block-container {{ padding-top: 3rem; padding-bottom: 2rem; }}
        
        div.stButton > button {{
            width: 100%; min-height: 45px; height: 100%; 
            white-space: normal; word-wrap: break-word;
            padding: 6px 8px; line-height: 1.25; 
            border-radius: 6px; border: 1px solid #e0e0e0;
            background-color: #ffffff; text-align: left; 
            display: flex; align-items: center;
            font-size: 0.8rem; box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            font-family: {font_css_rule} !important;
            transition: all 0.2s;
        }}
        div.stButton > button:hover {{
            border-color: #7c4dff; color: #7c4dff; 
            background-color: #f3f0ff;
            transform: translateY(-1px); 
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            z-index: 1;
        }}
        
        .group-header {{
            font-weight: 700; font-size: 0.9rem; color: #555;
            margin-top: 15px; margin-bottom: 5px; padding-bottom: 2px;
            border-bottom: 2px solid #f0f2f6;
            font-family: {font_css_rule} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. Gemini AI 引擎
# ==========================================
def get_valid_model():
    try:
        models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models: 
            if 'flash' in m.name: return m.name
        for m in models:
            if 'pro' in m.name: return m.name
        return 'gemini-pro'
    except: return 'gemini-pro'

def analyze_with_gemini(df, api_key):
    if not api_key: return None, "請先輸入 API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(get_valid_model())

        stats_info = {}
        for col in df.columns:
            n_unique = df[col].nunique()
            dtype = str(df[col].dtype)
            missing_rate = df[col].isnull().mean()
            col_profile = {"dtype": dtype, "unique": n_unique, "missing": f"{missing_rate:.1%}"}
            if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 0:
                try:
                    c_min, c_max = float(df[col].min()), float(df[col].max())
                    col_profile.update({"min": c_min, "max": c_max, "median": float(df[col].median())})
                    if (c_min > 190000 and c_max < 210012) or (c_min > 1900 and c_max < 2050 and n_unique < 50):
                        col_profile["hint"] = "TIME"
                except: pass
            try: col_profile["samples"] = df[col].dropna().astype(str).sample(min(5, len(df))).tolist()
            except: col_profile["samples"] = []
            stats_info[col] = col_profile

        summary = json.dumps(stats_info, ensure_ascii=False, indent=2)
        
        prompt = f"""
        <role>全方位數據視覺化架構師</role>
        <data>{summary}</data>
        <catalog>
        1. [AGG] 聚合類: "長條圖 (Bar)", "折線圖 (Line)", "面積圖 (Area)", "圓餅圖 (Pie)", "樹狀圖 (TreeMap)", "雷達圖 (Radar)", "漏斗圖 (Funnel)"
        2. [RAW] 原始分佈類: "直方圖 (Histogram)", "箱型圖 (Box Plot)", "散佈圖 (Scatter)"
        3. [CPLX] 複雜類: "雙軸組合圖 (Combo)"
        </catalog>
        <rules>
        生成 **20 個** 建議。
        1. **箱型圖 (Box Plot)** x2: 分析分佈與離群值。
        2. **散佈圖 (Scatter)** x2: 尋找關聯。
        3. **時間必備**: 遇到 `TIME` 必出 "折線圖" 或 "面積圖"。
        4. **高基數**: 分類>50 用 "長條圖 (Bar)" (Top 10)。
        </rules>
        <output>
        JSON Array: [{{ "group": "群組", "title": "標題", "chart_type": "類型", "x_col": "X", "y_col": "Y", "color_col": "Color", "sort": "desc/asc/none" }}]
        </output>
        """
        response = model.generate_content(prompt)
        json_str = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_str), None
    except Exception as e: return None, f"AI 分析失敗: {str(e)}"

# ==========================================
# 4. 資料載入
# ==========================================
@st.cache_data
def generate_demo_excel():
    rows = 600
    start = datetime(2023, 1, 1)
    dates = [start + timedelta(days=random.randint(0, 1000)) for _ in range(rows)]
    df = pd.DataFrame({
        '訂單': [f"ORD-{i}" for i in range(rows)],
        '年月份': [int(d.strftime('%Y%m')) for d in dates],
        '產品線': [random.choice(['旗艦機', '中階機', '入門機']) for _ in range(rows)],
        '區域': [random.choice(['北區', '中區', '南區', '東區']) for _ in range(rows)],
        '滿意度': np.random.randint(1, 10, rows),
        '產品評分': np.random.normal(7, 1.5, rows).clip(1, 10), 
        '單價': np.random.randint(5000, 30000, rows),
        '銷量': np.random.randint(1, 50, rows)
    })
    df.loc[0:10, '單價'] = 80000 
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False)
    return output.getvalue()

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or '日期' in col or 'Date' in col:
                try: df[col] = pd.to_datetime(df[col])
                except: pass
        return df
    except: return None

# ==========================================
# 5. 主介面 (V140 Two-Way Binding Architecture)
# ==========================================
st.title("✨ 作圖小工具 (Lyra V140)")

# --- Sidebar ---
with st.sidebar:
    st.header("1. 資料來源")
    st.session_state['gemini_api_key'] = st.text_input("🔑 Gemini Key", value=st.session_state.get('gemini_api_key',''), type="password")
    if st.button("✅ 連線"): 
        if st.session_state['gemini_api_key']: st.success("已連線")
    
    st.markdown("---")
    if st.button("🎲 生成測試資料"):
        st.download_button("📊 下載 Excel", generate_demo_excel(), "Demo_Data.xlsx")
    
    font_choice = st.selectbox("字體", ["Noto Sans TC (推薦)", "Microsoft JhengHei", "Arial"], index=0)
    inject_custom_css(font_choice)
    uploaded_files = st.file_uploader("上傳檔案", type=["xlsx", "csv"], accept_multiple_files=True)

df = None
CHART_TYPES = ["長條圖 (Bar)", "折線圖 (Line)", "雙軸組合圖 (Combo)", "圓餅圖 (Pie)", "樹狀圖 (TreeMap)", "散佈圖 (Scatter)", "箱型圖 (Box Plot)", "直方圖 (Histogram)", "雷達圖 (Radar)", "面積圖 (Area)", "漏斗圖 (Funnel)"]
RAW_DATA_CHARTS = ["箱型圖 (Box Plot)", "直方圖 (Histogram)", "散佈圖 (Scatter)"]
agg_funcs_list = ["總和 (Sum)", "平均 (Avg)", "最大值 (Max)", "計數 (Count)"]

if uploaded_files:
    file_map = {f.name: f for f in uploaded_files}
    with st.sidebar: selected_file_name = st.selectbox("選擇檔案", list(file_map.keys()))
    df = load_data(file_map[selected_file_name])
    
    if df is not None:
        # 重置索引：如果換了檔案，為了安全起見，重置某些索引
        if st.session_state.get('last_analyzed_file') != selected_file_name:
             # 不強制重置所有，但更新檔案標記
             pass 

        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        all_cols = df.columns.tolist()

        # --- Sidebar 設定區 (Callbacks) ---
        # 這些 callback 確保當使用者手動選的時候，idx 會更新
        def update_chart_idx(): st.session_state.chart_type_idx = safe_index(st.session_state.w_chart_type, CHART_TYPES)
        def update_x_idx(): st.session_state.x_col_idx = safe_index(st.session_state.w_x_col, all_cols)
        def update_y_idx(): st.session_state.y_col_idx = safe_index(st.session_state.w_y_col, num_cols)
        def update_y2_idx(): st.session_state.y_col_2_idx = safe_index(st.session_state.w_y2_col, num_cols)
        def update_color_idx(): st.session_state.color_col_idx = safe_index(st.session_state.w_color_col, ["(無)"]+all_cols)
        def update_agg_idx(): st.session_state.agg_func_idx = safe_index(st.session_state.w_agg, agg_funcs_list)

        with st.sidebar:
            st.markdown("---")
            st.header("2. 繪圖設定")
            
            # 1. Chart Type (Widget Key 加上 w_ 前綴，避免衝突)
            chart_type = st.selectbox(
                "圖表類型", CHART_TYPES, 
                index=st.session_state.chart_type_idx, 
                key='w_chart_type', on_change=update_chart_idx
            )
            
            # 2. X / Y Selection
            if chart_type == "雙軸組合圖 (Combo)":
                x_col = st.selectbox("X 軸", all_cols, index=st.session_state.x_col_idx, key='w_x_col', on_change=update_x_idx)
                y_col = st.selectbox("左軸", num_cols, index=st.session_state.y_col_idx, key='w_y_col', on_change=update_y_idx)
                y_col_2 = st.selectbox("右軸", num_cols, index=st.session_state.y_col_2_idx, key='w_y2_col', on_change=update_y2_idx)
            elif chart_type == "樹狀圖 (TreeMap)":
                treemap_path = st.multiselect("層級", cat_cols, default=st.session_state.treemap_path or cat_cols[:1], key='w_tree')
                st.session_state.treemap_path = treemap_path # 直接同步
                y_col = st.selectbox("數值", num_cols, index=st.session_state.y_col_idx, key='w_y_col', on_change=update_y_idx)
                x_col = all_cols[0] if all_cols else None # Dummy
            else:
                x_col = st.selectbox("X 軸", all_cols, index=st.session_state.x_col_idx, key='w_x_col', on_change=update_x_idx)
                y_col = st.selectbox("Y 軸", num_cols, index=st.session_state.y_col_idx, key='w_y_col', on_change=update_y_idx)
            
            # 3. Color & Agg
            if chart_type not in ["樹狀圖 (TreeMap)"]:
                color_col = st.selectbox("分組", ["(無)"] + all_cols, index=st.session_state.color_col_idx, key='w_color_col', on_change=update_color_idx)
            else: color_col = "(無)"
            
            agg_func = st.selectbox("計算", agg_funcs_list, index=st.session_state.agg_func_idx, key='w_agg', on_change=update_agg_idx)

        # =========================================================
        # 🚀 [Lyra V140] 圖表渲染引擎 (Protected)
        # =========================================================
        
        if df is not None:
            try:
                real_agg = {"總和 (Sum)": "sum", "平均 (Avg)": "mean", "最大值 (Max)": "max", "計數 (Count)": "count"}[agg_func]
                use_raw = chart_type in RAW_DATA_CHARTS
                
                # Data Preparation
                if use_raw: 
                    df_agg = df.copy()
                else:
                    cols = [c for c in [x_col, color_col] if c != "(無)"]
                    if not cols and chart_type != "樹狀圖 (TreeMap)":
                        df_agg = None
                    elif chart_type == "雙軸組合圖 (Combo)": 
                        df_agg = df.groupby(cols, as_index=False)[[y_col, y_col_2]].agg(real_agg)
                    elif chart_type == "樹狀圖 (TreeMap)": 
                        df_agg = df.groupby(treemap_path, as_index=False)[y_col].agg(real_agg) if treemap_path else None
                    else: 
                        df_agg = df.groupby(cols, as_index=False)[y_col].agg(real_agg)

                # Fix Numeric Date
                if df_agg is not None and chart_type != "直方圖 (Histogram)" and x_col in df_agg.columns and pd.api.types.is_numeric_dtype(df_agg[x_col]):
                    if 190000 < df_agg[x_col].mean() < 210012: df_agg[x_col] = df_agg[x_col].astype(str)

                # Sorting
                if chart_type in ["折線圖 (Line)", "面積圖 (Area)"] and df_agg is not None:
                    df_agg = df_agg.sort_values(by=x_col)
                elif not use_raw and df_agg is not None and chart_type not in ["樹狀圖 (TreeMap)", "雷達圖 (Radar)"]:
                     idx = st.session_state.sort_order_idx
                     if idx == 1: df_agg = df_agg.sort_values(by=y_col, ascending=False)
                     elif idx == 2: df_agg = df_agg.sort_values(by=y_col, ascending=True)

                # Plotting
                if df_agg is not None:
                    fig = None
                    p = {"data_frame": df_agg, "x": x_col if x_col in df_agg.columns else None, "title": f"{chart_type}"}
                    if color_col != "(無)": p["color"] = color_col
                    
                    if chart_type == "長條圖 (Bar)": fig = px.bar(**p, y=y_col, text_auto='.2s')
                    elif chart_type == "折線圖 (Line)": fig = px.line(**p, y=y_col, markers=True)
                    elif chart_type == "面積圖 (Area)": fig = px.area(**p, y=y_col)
                    elif chart_type == "漏斗圖 (Funnel)": fig = px.funnel(**p, y=y_col)
                    elif chart_type == "雙軸組合圖 (Combo)":
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        fig.add_trace(go.Bar(x=df_agg[x_col], y=df_agg[y_col], name=y_col, marker_color='#636EFA', opacity=0.8), secondary_y=False)
                        fig.add_trace(go.Scatter(x=df_agg[x_col], y=df_agg[y_col_2], name=y_col_2, mode='lines+markers', line=dict(color='#EF553B', width=3)), secondary_y=True)
                    elif chart_type == "圓餅圖 (Pie)": fig = px.pie(df_agg, values=y_col, names=x_col)
                    elif chart_type == "樹狀圖 (TreeMap)": fig = px.treemap(df_agg, path=treemap_path, values=y_col)
                    elif chart_type == "雷達圖 (Radar)": fig = px.line_polar(df_agg, r=y_col, theta=x_col, line_close=True, title="Radar")
                    elif chart_type == "直方圖 (Histogram)": fig = px.histogram(df_agg, x=x_col, color=p.get("color"))
                    elif chart_type == "箱型圖 (Box Plot)": fig = px.box(df_agg, x=x_col, y=y_col, color=p.get("color"))
                    elif chart_type == "散佈圖 (Scatter)": fig = px.scatter(df_agg, x=x_col, y=y_col, color=p.get("color"))

                    if fig:
                        fig.update_layout(template="plotly_white", height=500, margin=dict(t=30, b=50, l=50, r=50), font=dict(family=font_choice.split(',')[0].strip("'"), size=16), hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("請選擇正確的 X/Y 軸欄位以生成圖表")

            except Exception as e: 
                # [V140 Fix] 顯示更詳細的錯誤，避免只顯示 None
                st.error(f"繪圖發生錯誤: {repr(e)}")

        # =========================================================
        # 🚀 [Lyra V140] AI 按鈕區 (完全透過 Index 同步)
        # =========================================================
        st.markdown("---")
        st.subheader("🤖 AI 戰略分析面板")
        st.caption("👇 捲動挑選 (已修復同步問題)")

        with st.container(height=300):
            if st.session_state.get('gemini_api_key'):
                if 'last_analyzed_file' not in st.session_state or st.session_state['last_analyzed_file'] != selected_file_name:
                    with st.spinner("🧠 正在掃描全域數據特徵..."):
                        insights, error_msg = analyze_with_gemini(df, st.session_state['gemini_api_key'])
                        if not error_msg:
                            st.session_state['ai_insights'] = insights
                            st.session_state['last_analyzed_file'] = selected_file_name
                
                if st.session_state.get('ai_insights'):
                    insights = st.session_state['ai_insights']
                    groups = sorted(list(set(ins['group'] for ins in insights)))
                    
                    for group_name in groups:
                        st.markdown(f"<div class='group-header'>{group_name}</div>", unsafe_allow_html=True)
                        cols = st.columns(5)
                        group_insights = [ins for ins in insights if ins['group'] == group_name]
                        for i, insight in enumerate(group_insights):
                            with cols[i % 5]:
                                if st.button(f"📊 {insight['title']}", key=f"btn_{group_name}_{i}", use_container_width=True):
                                    raw = insight.get('chart_type', '')
                                    matched = "長條圖 (Bar)"
                                    for t in CHART_TYPES:
                                        if any(k.strip("()") in raw for k in t.split(' ') if len(k)>2):
                                            matched = t; break
                                    
                                    # [V140 Fix] 這裡只更新 Index，不碰 Widget Key
                                    st.session_state.chart_type_idx = safe_index(matched, CHART_TYPES)
                                    st.session_state.x_col_idx = safe_index(insight.get('x_col'), all_cols)
                                    st.session_state.y_col_idx = safe_index(insight.get('y_col'), num_cols)
                                    st.session_state.color_col_idx = safe_index(insight.get('color_col'), ["(無)"]+all_cols)
                                    
                                    sort = insight.get('sort', 'none')
                                    st.session_state.sort_order_idx = 1 if sort=='desc' else 2 if sort=='asc' else 0
                                    
                                    if matched == "樹狀圖 (TreeMap)" and insight.get('x_col'):
                                        st.session_state.treemap_path = [insight.get('x_col')]
                                    
                                    st.rerun()
            else:
                st.warning("請在左側輸入 Gemini API Key 以啟動分析面板")
