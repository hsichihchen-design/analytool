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
st.set_page_config(page_title="作圖小工具 V50.1 (Lyra Fix)", layout="wide", page_icon="✨")

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
# 2. 核心功能：Gemini AI 分析引擎 (Lyra V2 Enhanced)
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

        # --- [Lyra Strategy 1] Python 端資料特徵工程 (The Detective) ---
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

            # 針對數值型 (int/float) 增加統計特徵
            if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 0:
                try:
                    c_min = float(df[col].min())
                    c_max = float(df[col].max())
                    col_profile["min"] = c_min
                    col_profile["max"] = c_max
                    col_profile["median"] = float(df[col].median())
                    
                    # [關鍵判斷] 偵測是否為「偽裝成數字的時間」 (如 202401)
                    # 規則：介於 199001 ~ 203012 之間 (年月) 或 1990 ~ 2030 (年)
                    is_yyyymm = (c_min > 190000 and c_max < 210012)
                    is_yyyy = (c_min > 1900 and c_max < 2050 and n_unique < 50)
                    if is_yyyymm or is_yyyy:
                        col_profile["semantic_hint"] = "TIME_SERIES (Treat as String/Category)"
                except:
                    pass
            
            try:
                clean_samples = df[col].dropna().astype(str).sample(min(5, len(df))).tolist()
            except:
                clean_samples = []
            col_profile["samples"] = clean_samples
            
            stats_info[col] = col_profile

        columns_summary = json.dumps(stats_info, ensure_ascii=False, indent=2)
        # ---------------------------------------------

        # --- [Lyra Strategy 2] 語意推論 Prompt (The Brain) ---
        prompt = f"""
        <role>
        你是一位精通商業分析與 Plotly 視覺化的專家。你的任務是解決「數值誤判」問題並提供最佳圖表建議。
        </role>

        <data_profile>
        {columns_summary}
        </data_profile>

        <critical_instruction>
        請特別注意 `semantic_hint` 標記。若欄位標記為 "TIME_SERIES" (例如 202401, 202402)，
        它雖然是數字，但**必須視為時間分類 (Category)**。
        1. **X軸處理**: 它是畫 Trend (折線/長條) 的最佳選擇。
        2. **禁止事項**: 絕對不要對此欄位做直方圖 (Histogram) 或加總 (Sum)。
        3. **排序**: 通常維持自然順序 (asc)。
        </critical_instruction>

        <output_format>
        請回傳純 JSON Array (不要 Markdown):
        [
          {{
            "group": "圖表分類 (趨勢/排行/佔比/關聯)",
            "title": "標題 (Max 10字)",
            "chart_type": "圖表類型 (長條圖/折線圖/雙軸組合圖/圓餅圖/樹狀圖/散佈圖/箱型圖)",
            "x_col": "欄位名",
            "y_col": "欄位名",
            "color_col": "欄位名 (若不適合分組請填 null)",
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
# 3. 輔助函式
# ==========================================
def get_manual_content():
    return """
# 📊 作圖小工具 V50.1 (Lyra Fix)
已針對「數值型日期 (如 202401)」造成的圖表斷層與標籤錯誤進行核心修復。
AI 現在能識別年月格式，繪圖引擎會自動將其轉為文字標籤，確保圖表連續且美觀。
    """

@st.cache_data
def generate_demo_excel():
    rows = 500
    start_date = datetime(2023, 1, 1)
    # 模擬 202301 ~ 202512 的資料
    dates = [start_date + timedelta(days=random.randint(0, 1000)) for _ in range(rows)]
    df = pd.DataFrame({
        '訂單編號': [f"ORD-{i}" for i in range(rows)],
        '年月份 (容易出錯)': [int(d.strftime('%Y%m')) for d in dates], # 202301 (int)
        '產品類別': [random.choice(['手機', '筆電', '配件']) for _ in range(rows)],
        '銷售金額': np.random.randint(1000, 20000, rows)
    })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'): df = pd.read_csv(file)
        else: df = pd.read_excel(file)
        # 日期轉換
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
st.title("✨ 作圖小工具 (Lyra Fix)")

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
    with st.expander("📥 範例資料"):
        if st.button("🎲 生成測試資料 (含年月格式)"):
            st.download_button("📊 下載 Excel", generate_demo_excel(), "Demo_Data.xlsx")

    font_choice = st.selectbox("字體", ["Noto Sans TC (推薦)", "Microsoft JhengHei", "Arial"], index=0)
    inject_custom_css(font_choice)
    uploaded_files = st.file_uploader("上傳 Excel/CSV", type=["xlsx", "csv"], accept_multiple_files=True)

df = None
all_cols, num_cols, cat_cols = [], [], []

if uploaded_files:
    file_map = {f.name: f for f in uploaded_files}
    with st.sidebar: selected_file_name = st.selectbox("選擇檔案", list(file_map.keys()))
    df = load_data(file_map[selected_file_name])
    
    if df is not None:
        # 時間粒度處理 (略過，維持原樣)
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        for col in date_cols:
            df[f"{col}(YM)"] = df[col].dt.strftime('%Y-%m')

        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        all_cols = df.columns.tolist()
        
        chart_types_list = ["長條圖 (Bar)", "折線圖 (Line)", "雙軸組合圖 (Combo)", "圓餅圖 (Pie)", "樹狀圖 (TreeMap)", "散佈圖 (Scatter)", "箱型圖 (Box Plot)", "直方圖 (Histogram)", "雷達圖 (Radar)"]
        agg_funcs_list = ["總和 (Sum)", "平均 (Avg)", "最大值 (Max)", "計數 (Count)"]
        
        # --- AI 分析區塊 ---
        st.markdown("---")
        st.subheader("🤖 Lyra 戰略建議")
        
        if st.session_state['gemini_api_key']:
            if 'last_analyzed_file' not in st.session_state or st.session_state['last_analyzed_file'] != selected_file_name:
                with st.spinner("🧠 正在掃描資料特徵與異常值..."):
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
                                # 更新 Session State
                                c_type = insight.get('chart_type', '長條圖 (Bar)')
                                if c_type in chart_types_list: 
                                    st.session_state['chart_type_idx'] = chart_types_list.index(c_type)
                                    st.session_state['chart_type_box'] = c_type
                                
                                # 自動填入欄位
                                def sync(key, val, candidates):
                                    if val in candidates: st.session_state[f"{key}_idx"], st.session_state[f"{key}_box"] = candidates.index(val), val
                                
                                sync('x_col', insight.get('x_col'), all_cols)
                                sync('y_col', insight.get('y_col'), num_cols)
                                sync('color_col', insight.get('color_col'), ["(無)"]+all_cols)
                                st.rerun()

        # --- 繪圖設定區 ---
        with st.sidebar:
            st.markdown("---")
            st.header("2. 繪圖設定")
            chart_type = st.selectbox("圖表類型", chart_types_list, index=st.session_state['chart_type_idx'], key='chart_type_box')
            
            # 簡化版 UI 邏輯
            if chart_type == "雙軸組合圖 (Combo)":
                x_col = st.selectbox("X 軸", all_cols, index=st.session_state['x_col_idx'], key='x_col_box')
                y_col = st.selectbox("左軸數值", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                y_col_2 = st.selectbox("右軸數值", num_cols, index=st.session_state['y_col_2_idx'], key='y_col_2_box')
                agg_func = st.selectbox("計算", agg_funcs_list, index=st.session_state['agg_func_idx'], key='agg_func_box')
                color_col = st.selectbox("分組", ["(無)"] + all_cols, index=st.session_state['color_col_idx'], key='color_col_box')
            else:
                x_col = st.selectbox("X 軸", all_cols, index=st.session_state['x_col_idx'], key='x_col_box')
                y_col = st.selectbox("Y 軸", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                agg_func = st.selectbox("計算", agg_funcs_list, index=st.session_state['agg_func_idx'], key='agg_func_box')
                color_col = st.selectbox("分組", ["(無)"] + all_cols, index=st.session_state['color_col_idx'], key='color_col_box')

        # --- [Lyra Strategy 3] 渲染層強制轉型與繪圖 (The Rendering Engine) ---
        if df is not None:
            try:
                # 準備 aggregation map
                agg_map = {"總和 (Sum)": "sum", "平均 (Avg)": "mean", "最大值 (Max)": "max", "計數 (Count)": "count"}
                real_agg = agg_map[agg_func]
                
                # 準備分組欄位
                grp_cols = [x_col]
                if color_col != "(無)": grp_cols.append(color_col)
                
                # 執行 Groupby
                if chart_type == "雙軸組合圖 (Combo)":
                    df_agg = df.groupby(grp_cols, as_index=False)[[y_col, y_col_2]].agg(real_agg)
                else:
                    df_agg = df.groupby(grp_cols, as_index=False)[y_col].agg(real_agg)

                # ========================================================
                # 🚀 Lyra Core Fix: X軸 數值型日期 強制轉型 (String Casting)
                # ========================================================
                # 判斷邏輯：如果是數字類型，且平均值 > 1900 (年份) 且不是純 ID (ID通常不會拿來做 X 軸分組統計，除非是Bar)
                # 這裡直接粗暴一點：只要是用來做 X 軸的「數值」，而且看起來像年份或年月，就轉字串。
                if pd.api.types.is_numeric_dtype(df_agg[x_col]):
                    col_mean = df_agg[x_col].mean()
                    # 1990 ~ 2100 (年份) 或 199001 ~ 210012 (年月)
                    if (1900 < col_mean < 2100) or (190000 < col_mean < 210012):
                        df_agg[x_col] = df_agg[x_col].astype(str)
                        st.toast(f"🛡️ 已自動將 X 軸 '{x_col}' 轉為文字模式，以避免圖表斷層。", icon="✨")
                # ========================================================

                # 開始繪圖 (Plotly)
                fig = None
                common_params = {
                    "data_frame": df_agg, 
                    "x": x_col, 
                    "title": f"{x_col} 分析報表"
                }
                if color_col != "(無)": common_params["color"] = color_col

                if chart_type == "長條圖 (Bar)":
                    fig = px.bar(**common_params, y=y_col, text_auto='.2s')
                elif chart_type == "折線圖 (Line)":
                    fig = px.line(**common_params, y=y_col, markers=True)
                elif chart_type == "雙軸組合圖 (Combo)":
                    # 建立雙軸物件
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    # 左軸 (Bar)
                    fig.add_trace(
                        go.Bar(x=df_agg[x_col], y=df_agg[y_col], name=y_col, marker_color='#636EFA', opacity=0.8),
                        secondary_y=False
                    )
                    # 右軸 (Line)
                    fig.add_trace(
                        go.Scatter(x=df_agg[x_col], y=df_agg[y_col_2], name=y_col_2, mode='lines+markers', line=dict(color='#EF553B', width=3)),
                        secondary_y=True
                    )
                    fig.update_layout(title=f"{y_col} vs {y_col_2}")
                elif chart_type == "圓餅圖 (Pie)":
                    fig = px.pie(df_agg, values=y_col, names=x_col, title=f"{x_col} 佔比")
                
                # 其他圖表類型略 (可依此類推)...
                if not fig and chart_type == "直方圖 (Histogram)":
                    fig = px.histogram(df, x=x_col, color=color_col if color_col!="(無)" else None)

                if fig:
                    # 優化 Layout
                    fig.update_layout(
                        template="plotly_white", 
                        height=600,
                        font=dict(family=font_choice.split(',')[0].strip("'"), size=16),
                        hovermode="x unified"
                    )
                    # 如果 X 軸轉成了字串，Plotly 預設就是 Category，斷層自然消失
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"繪圖錯誤: {e}")
