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
# 0. 初始化 Session State
# ==========================================
default_states = {
    'chart_type_idx': 0, 
    'x_col_idx': 0,
    'y_col_idx': 0,
    'y_col_2_idx': 0,
    'color_col_idx': 0,
    'facet_col_idx': 0,
    'agg_func_idx': 0,
    'sort_order_idx': 0,
    'ref_line_val': 0.0,
    'treemap_path': [],
    'gemini_api_key': '',
    'ai_insights': [],
    'last_analyzed_file': ''
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==========================================
# 1. 全域設定與 CSS
# ==========================================
st.set_page_config(page_title="作圖小工具 V80 (Lyra Time-Keeper)", layout="wide", page_icon="✨")

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
        .stDownloadButton button {{ width: 100%; border-color: #4CAF50; color: #4CAF50; }}
        div.stButton > button {{
            width: 100%; min-height: 45px; height: 100%; white-space: normal; word-wrap: break-word;
            padding: 4px 6px; line-height: 1.2; border-radius: 4px; border: 1px solid #ddd;
            background-color: #ffffff; text-align: left; display: flex; align-items: center;
            font-size: 0.8rem; box-shadow: 0 1px 1px rgba(0,0,0,0.05);
            font-family: {font_css_rule} !important;
        }}
        div.stButton > button:hover {{
            border-color: #7c4dff; color: #7c4dff; background-color: #f8f5ff;
            transform: translateY(-1px); z-index: 1;
        }}
        .group-header {{
            font-weight: 700; font-size: 0.9rem; color: #666;
            margin-top: 15px; margin-bottom: 5px; padding-bottom: 2px;
            border-bottom: 1px solid #eee;
            font-family: {font_css_rule} !important;
        }}
        .block-container {{ padding-top: 3.5rem; }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 核心功能：Gemini AI 分析引擎
# ==========================================

def get_valid_model():
    try:
        models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models: 
            if 'flash' in m.name: return m.name
        for m in models:
            if 'pro' in m.name: return m.name
        return 'gemini-pro'
    except:
        return 'gemini-pro'

def analyze_with_gemini(df, api_key):
    if not api_key:
        return None, "請先輸入 API Key。"

    try:
        genai.configure(api_key=api_key)
        model_name = get_valid_model()
        model = genai.GenerativeModel(model_name)

        # --- [Lyra Detective] 資料掃描 ---
        stats_info = {}
        for col in df.columns:
            n_unique = df[col].nunique()
            dtype = str(df[col].dtype)
            missing_rate = df[col].isnull().mean()
            col_profile = {
                "dtype": dtype,
                "unique_count": n_unique,
                "missing_rate": f"{missing_rate:.1%}"
            }
            if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 0:
                try:
                    c_min = float(df[col].min())
                    c_max = float(df[col].max())
                    col_profile["min"] = c_min
                    col_profile["max"] = c_max
                    col_profile["median"] = float(df[col].median())
                    is_yyyymm = (c_min > 190000 and c_max < 210012)
                    is_yyyy = (c_min > 1900 and c_max < 2050 and n_unique < 50)
                    if is_yyyymm or is_yyyy:
                        col_profile["semantic_hint"] = "TIME_SERIES"
                except: pass
            
            try: clean_samples = df[col].dropna().astype(str).sample(min(5, len(df))).tolist()
            except: clean_samples = []
            col_profile["samples"] = clean_samples
            stats_info[col] = col_profile

        columns_summary = json.dumps(stats_info, ensure_ascii=False, indent=2)
        
        # --- [Lyra Architect Prompt] ---
        prompt = f"""
        <role>
        你是一位數據視覺化架構師。請根據數據特性，分配不同的分析任務，挖掘多維度 Insight。
        </role>

        <data_profile>
        {columns_summary}
        </data_profile>

        <chart_catalog>
        標準名稱 (請嚴格遵守):
        1. [AGG] 聚合類: "長條圖 (Bar)", "折線圖 (Line)", "面積圖 (Area)", "圓餅圖 (Pie)", "樹狀圖 (TreeMap)", "雷達圖 (Radar)", "漏斗圖 (Funnel)"
        2. [RAW] 原始分佈類: "直方圖 (Histogram)", "箱型圖 (Box Plot)", "散佈圖 (Scatter)"
        3. [CPLX] 複雜類: "雙軸組合圖 (Combo)"
        </chart_catalog>

        <mandatory_requirements>
        請生成 **20 個** 建議，且必須滿足以下配額 (Diversity Quota)：
        1. **至少 2 個 "箱型圖 (Box Plot)"**: 用於分析數值分佈與離群值。
        2. **至少 2 個 "散佈圖 (Scatter)"**: 用於尋找變數關聯。
        3. **時間序列必備**: 遇到 `TIME_SERIES`，必須出 "折線圖" 或 "面積圖"。
        4. **高基數處理**: 若分類 > 50 種，建議 "長條圖 (Bar)" 並標註 (Top 10)。
        </mandatory_requirements>

        <output_format>
        回傳純 JSON Array:
        [
          {{
            "group": "群組名稱",
            "title": "標題 (Max 10字)",
            "chart_type": "必須完全符合 chart_catalog 的中文名稱",
            "x_col": "欄位名",
            "y_col": "欄位名",
            "color_col": "欄位名 (可null)",
            "sort": "desc/asc/none"
          }}
        ]
        </output_format>
        """

        response = model.generate_content(prompt)
        json_str = response.text.strip()
        if json_str.startswith("```json"): json_str = json_str[7:]
        if json_str.startswith("```"): json_str = json_str[3:]
        if json_str.endswith("```"): json_str = json_str[:-3]
            
        insights = json.loads(json_str)
        return insights, None

    except Exception as e:
        return None, f"AI 分析失敗: {str(e)}"

# ==========================================
# 3. 輔助與資料載入
# ==========================================
def get_manual_content():
    return """
# 📊 作圖小工具 V80 (Lyra Time-Keeper)
## 核心修復
1. **折線圖亂跑修復**：強制加入時序鎖定機制，確保折線圖與面積圖永遠依照 X 軸（時間）順序由左至右繪製，不再受數值排序影響。
2. **箱型圖/散佈圖優化**：保留原始數據分佈，正確顯示統計特徵。
3. **模糊匹配**：自動修正 AI 回傳的圖表名稱誤差。
    """

@st.cache_data
def generate_demo_excel():
    rows = 600
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 1000)) for _ in range(rows)]
    df = pd.DataFrame({
        '訂單編號': [f"ORD-{i}" for i in range(rows)],
        '年月份': [int(d.strftime('%Y%m')) for d in dates],
        '產品線': [random.choice(['旗艦機', '中階機', '入門機']) for _ in range(rows)],
        '區域': [random.choice(['北區', '中區', '南區', '東區']) for _ in range(rows)],
        '客戶滿意度': np.random.randint(1, 10, rows),
        '產品評分': np.random.normal(7, 1.5, rows).clip(1, 10), 
        '物流評分': np.random.normal(8, 1, rows).clip(1, 10),
        '單價': np.random.randint(5000, 30000, rows),
        '銷量': np.random.randint(1, 50, rows),
        '折扣率': np.random.uniform(0.8, 1.0, rows)
    })
    df.loc[0:10, '單價'] = 80000 
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'): df = pd.read_csv(file)
        else: df = pd.read_excel(file)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or '日期' in col or 'Date' in col:
                try: df[col] = pd.to_datetime(df[col])
                except: pass
        return df
    except Exception as e:
        st.error(f"檔案讀取失敗: {e}")
        return None

# ==========================================
# 4. 主介面
# ==========================================
st.title("✨ 作圖小工具 (Lyra V80)")

with st.sidebar:
    st.header("1. 資料來源")
    st.session_state['gemini_api_key'] = st.text_input("🔑 Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    
    if st.button("✅ 驗證 API Key", use_container_width=True):
        if st.session_state['gemini_api_key']:
            try:
                genai.configure(api_key=st.session_state['gemini_api_key'])
                list(genai.list_models())
                st.success(f"連線成功！")
            except Exception as e: st.error(f"連線失敗: {e}")

    st.markdown("---")
    with st.expander("📥 範例資料 (含離群值)"):
        if st.button("🎲 生成 V80 測試資料"):
            st.download_button("📊 下載 Excel", generate_demo_excel(), "Demo_Data.xlsx")

    font_choice = st.selectbox("字體", ["Noto Sans TC (推薦)", "Microsoft JhengHei", "Arial"], index=0)
    inject_custom_css(font_choice)
    uploaded_files = st.file_uploader("上傳 Excel/CSV", type=["xlsx", "csv"], accept_multiple_files=True)

df = None
all_cols, num_cols, cat_cols = [], [], []

CHART_TYPES = [
    "長條圖 (Bar)", "折線圖 (Line)", "雙軸組合圖 (Combo)", "圓餅圖 (Pie)", 
    "樹狀圖 (TreeMap)", "散佈圖 (Scatter)", "箱型圖 (Box Plot)", 
    "直方圖 (Histogram)", "雷達圖 (Radar)", "面積圖 (Area)", "漏斗圖 (Funnel)"
]
RAW_DATA_CHARTS = ["箱型圖 (Box Plot)", "直方圖 (Histogram)", "散佈圖 (Scatter)"]

if uploaded_files:
    file_map = {f.name: f for f in uploaded_files}
    with st.sidebar: selected_file_name = st.selectbox("選擇檔案", list(file_map.keys()))
    df = load_data(file_map[selected_file_name])
    
    if df is not None:
        # 時間粒度
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        for col in date_cols:
            df[f"{col}(YM)"] = df[col].dt.strftime('%Y-%m')

        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        all_cols = df.columns.tolist()
        agg_funcs_list = ["總和 (Sum)", "平均 (Avg)", "最大值 (Max)", "計數 (Count)"]
        
        # --- AI 分析區塊 ---
        st.markdown("---")
        st.subheader("🤖 Lyra 深度分析建議")
        
        if st.session_state['gemini_api_key']:
            if 'last_analyzed_file' not in st.session_state or st.session_state['last_analyzed_file'] != selected_file_name:
                with st.spinner("🧠 正在構建分析矩陣..."):
                    insights, error_msg = analyze_with_gemini(df, st.session_state['gemini_api_key'])
                    if not error_msg:
                        st.session_state['ai_insights'] = insights
                        st.session_state['last_analyzed_file'] = selected_file_name
            
            if st.session_state.get('ai_insights'):
                insights = st.session_state['ai_insights']
                groups = sorted(list(set(ins['group'] for ins in insights)))
                for group_name in groups:
                    st.markdown(f"<div class='group-header'>{group_name}</div>", unsafe_allow_html=True)
                    cols = st.columns(7) 
                    group_insights = [ins for ins in insights if ins['group'] == group_name]
                    for i, insight in enumerate(group_insights):
                        with cols[i % 7]:
                            if st.button(insight['title'], key=f"btn_{group_name}_{i}"):
                                # 模糊匹配
                                raw_type = insight.get('chart_type', '')
                                matched_type = "長條圖 (Bar)"
                                for standard_type in CHART_TYPES:
                                    keywords = standard_type.split(' ')
                                    if any(k.strip("()") in raw_type for k in keywords if len(k)>2):
                                        matched_type = standard_type
                                        break
                                
                                st.session_state['chart_type_idx'] = CHART_TYPES.index(matched_type)
                                st.session_state['chart_type_box'] = matched_type
                                
                                def sync(key, val, candidates):
                                    if val in candidates: st.session_state[f"{key}_idx"], st.session_state[f"{key}_box"] = candidates.index(val), val
                                    else: st.session_state[f"{key}_idx"] = 0
                                
                                sync('x_col', insight.get('x_col'), all_cols)
                                sync('y_col', insight.get('y_col'), num_cols)
                                sync('color_col', insight.get('color_col'), ["(無)"]+all_cols)
                                
                                sort_str = insight.get('sort', 'none')
                                if sort_str == 'desc': st.session_state['sort_order_idx'] = 1
                                elif sort_str == 'asc': st.session_state['sort_order_idx'] = 2
                                else: st.session_state['sort_order_idx'] = 0
                                
                                if matched_type == "樹狀圖 (TreeMap)" and insight.get('x_col'):
                                     st.session_state['treemap_path'] = [insight.get('x_col')]
                                     st.session_state['treemap_box'] = [insight.get('x_col')]
                                
                                st.rerun()

        # --- 繪圖設定區 ---
        with st.sidebar:
            st.markdown("---")
            st.header("2. 繪圖設定")
            chart_type = st.selectbox("圖表類型", CHART_TYPES, index=st.session_state['chart_type_idx'], key='chart_type_box')
            
            # UI
            if chart_type == "雙軸組合圖 (Combo)":
                x_col = st.selectbox("X 軸", all_cols, index=st.session_state['x_col_idx'], key='x_col_box')
                y_col = st.selectbox("左軸數值", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                y_col_2 = st.selectbox("右軸數值", num_cols, index=st.session_state['y_col_2_idx'], key='y_col_2_box')
                agg_func = st.selectbox("計算", agg_funcs_list, index=st.session_state['agg_func_idx'], key='agg_func_box')
                color_col = st.selectbox("分組", ["(無)"] + all_cols, index=st.session_state['color_col_idx'], key='color_col_box')
            elif chart_type == "樹狀圖 (TreeMap)":
                default_tree_path = st.session_state['treemap_path'] if st.session_state['treemap_path'] else (cat_cols[:2] if len(cat_cols)>=2 else cat_cols[:1])
                treemap_path = st.multiselect("層級結構", cat_cols, default=default_tree_path, key='treemap_box')
                y_col = st.selectbox("數值大小", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                color_col = st.selectbox("顏色依據", ["(無)"] + num_cols + cat_cols, index=st.session_state['color_col_idx'], key='color_col_box')
                agg_func = st.selectbox("計算", agg_funcs_list, index=st.session_state['agg_func_idx'], key='agg_func_box')
            elif chart_type == "雷達圖 (Radar)":
                x_col = st.selectbox("維度 (Label)", cat_cols, index=st.session_state['x_col_idx'], key='x_col_box')
                y_col = st.selectbox("數值 (Value)", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                color_col = st.selectbox("分組", ["(無)"] + all_cols, index=st.session_state['color_col_idx'], key='color_col_box')
                agg_func = st.selectbox("計算", agg_funcs_list, index=st.session_state['agg_func_idx'], key='agg_func_box')
            else:
                x_col = st.selectbox("X 軸", all_cols, index=st.session_state['x_col_idx'], key='x_col_box')
                y_col = st.selectbox("Y 軸 (數值)", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                agg_func = st.selectbox("計算", agg_funcs_list, index=st.session_state['agg_func_idx'], key='agg_func_box')
                color_col = st.selectbox("分組", ["(無)"] + all_cols, index=st.session_state['color_col_idx'], key='color_col_box')

        # --- [Lyra Rendering Engine V80] ---
        if df is not None:
            try:
                agg_map = {"總和 (Sum)": "sum", "平均 (Avg)": "mean", "最大值 (Max)": "max", "計數 (Count)": "count"}
                real_agg = agg_map[agg_func]
                
                # 數據分流
                use_raw_data = chart_type in RAW_DATA_CHARTS
                
                if use_raw_data:
                    df_agg = df.copy()
                else:
                    if chart_type == "雙軸組合圖 (Combo)":
                         grp_cols = [x_col]
                         if color_col != "(無)": grp_cols.append(color_col)
                         df_agg = df.groupby(grp_cols, as_index=False)[[y_col, y_col_2]].agg(real_agg)
                    elif chart_type == "樹狀圖 (TreeMap)":
                         if not treemap_path: df_agg = None
                         else: df_agg = df.groupby(treemap_path, as_index=False)[y_col].agg(real_agg)
                    elif chart_type == "雷達圖 (Radar)":
                         grp_cols = [x_col]
                         if color_col != "(無)": grp_cols.append(color_col)
                         df_agg = df.groupby(grp_cols, as_index=False)[y_col].agg(real_agg)
                    else:
                        grp_cols = [x_col]
                        if color_col != "(無)": grp_cols.append(color_col)
                        df_agg = df.groupby(grp_cols, as_index=False)[y_col].agg(real_agg)

                    # [Lyra Logic] X 軸數值型日期轉字串 (防斷層)
                    if df_agg is not None and x_col in df_agg.columns and pd.api.types.is_numeric_dtype(df_agg[x_col]):
                        col_mean = df_agg[x_col].mean()
                        if (1900 < col_mean < 2100) or (190000 < col_mean < 210012):
                            df_agg[x_col] = df_agg[x_col].astype(str)

                # [Lyra V80 Feature] 時序鎖定 (Chronological Lock)
                # 即使上方面板或 AI 指定了排序，若為折線/面積圖，強制依照 X 軸排序
                if chart_type in ["折線圖 (Line)", "面積圖 (Area)"] and df_agg is not None:
                    df_agg = df_agg.sort_values(by=x_col, ascending=True)
                # 其他聚合圖表依設定排序
                elif not use_raw_data and df_agg is not None and chart_type not in ["樹狀圖 (TreeMap)", "雷達圖 (Radar)"]:
                    sort_idx = st.session_state['sort_order_idx']
                    if sort_idx == 1: df_agg = df_agg.sort_values(by=y_col, ascending=False)
                    elif sort_idx == 2: df_agg = df_agg.sort_values(by=y_col, ascending=True)

                # 開始繪圖
                if df_agg is not None:
                    fig = None
                    common_params = {"data_frame": df_agg, "x": x_col if x_col in df_agg.columns else None, "title": f"{chart_type}: {x_col if x_col else ''}"}
                    if color_col != "(無)" and color_col in df_agg.columns: common_params["color"] = color_col

                    if chart_type == "長條圖 (Bar)":
                        fig = px.bar(**common_params, y=y_col, text_auto='.2s')
                    elif chart_type == "折線圖 (Line)":
                        fig = px.line(**common_params, y=y_col, markers=True)
                    elif chart_type == "面積圖 (Area)":
                        fig = px.area(**common_params, y=y_col)
                    elif chart_type == "漏斗圖 (Funnel)":
                        fig = px.funnel(**common_params, y=y_col)
                    elif chart_type == "雙軸組合圖 (Combo)":
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        fig.add_trace(go.Bar(x=df_agg[x_col], y=df_agg[y_col], name=y_col, marker_color='#636EFA', opacity=0.8), secondary_y=False)
                        fig.add_trace(go.Scatter(x=df_agg[x_col], y=df_agg[y_col_2], name=y_col_2, mode='lines+markers', line=dict(color='#EF553B', width=3)), secondary_y=True)
                        fig.update_layout(title=f"{y_col} vs {y_col_2}")
                    elif chart_type == "圓餅圖 (Pie)":
                        fig = px.pie(df_agg, values=y_col, names=x_col, title=f"{x_col} 佔比")
                    elif chart_type == "樹狀圖 (TreeMap)":
                        fig = px.treemap(df_agg, path=treemap_path, values=y_col, color=color_col if color_col!="(無)" else y_col, title="層級分析")
                    elif chart_type == "雷達圖 (Radar)":
                         fig = px.line_polar(df_agg, r=y_col, theta=x_col, line_close=True, color=color_col if color_col != "(無)" else None, title="雷達分析")
                         fig.update_traces(fill='toself')
                    elif chart_type == "直方圖 (Histogram)":
                        fig = px.histogram(df_agg, x=x_col, color=color_col if color_col!="(無)" else None, title=f"{x_col} 分佈")
                    elif chart_type == "箱型圖 (Box Plot)":
                        fig = px.box(df_agg, x=x_col, y=y_col, color=color_col if color_col!="(無)" else None, title=f"{y_col} 分佈 (by {x_col})")
                    elif chart_type == "散佈圖 (Scatter)":
                        fig = px.scatter(df_agg, x=x_col, y=y_col, color=color_col if color_col!="(無)" else None, title=f"{x_col} vs {y_col}")

                    if fig:
                        fig.update_layout(template="plotly_white", height=600, font=dict(family=font_choice.split(',')[0].strip("'"), size=16), hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"繪圖錯誤: {e}")
