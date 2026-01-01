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
    'gemini_api_key': ''
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==========================================
# 1. 全域設定與 CSS (極致緊湊版)
# ==========================================
st.set_page_config(page_title="作圖小工具 V37.1", layout="wide", page_icon="✨")

def inject_custom_css(font_family):
    st.markdown(f"""
    <style>
        html, body, [class*="css"] {{
            font-family: '{font_family}', 'Microsoft JhengHei', sans-serif !important;
        }}
        .stDownloadButton button {{ width: 100%; border-color: #4CAF50; color: #4CAF50; }}
        
        /* 極致緊湊按鈕 */
        div.stButton > button {{
            width: 100%; 
            min-height: 45px;  /* 壓縮高度 */
            height: 100%;
            padding: 6px 10px; /* 減少內距 */
            line-height: 1.2; 
            border-radius: 6px; 
            border: 1px solid #e6e6e6;
            background-color: #ffffff; 
            text-align: left; 
            display: flex; 
            align-items: center;
            font-size: 0.9rem; /* 字體微調 */
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        }}
        div.stButton > button:hover {{
            border-color: #7c4dff; 
            color: #7c4dff; 
            background-color: #f8f5ff;
            transform: translateY(-1px);
            box-shadow: 0 3px 5px rgba(0,0,0,0.08);
        }}
        
        /* 壓縮群組標題間距 */
        .group-header {{
            font-weight: 700; 
            font-size: 1rem; 
            color: #555;
            margin-top: 15px; 
            margin-bottom: 5px; 
            padding-bottom: 3px;
            border-bottom: 1px solid #eee;
        }}
        
        /* 隱藏不必要的空白 */
        .block-container {{ padding-top: 2rem; }}
        
        [data-testid="stSidebar"] [data-testid="stTextInput"] input {{
            border-color: #7c4dff;
        }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 核心功能：Gemini AI 分析引擎
# ==========================================

def get_valid_model():
    """動態取得可用模型，避免 404"""
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

        stats_info = {}
        for col in df.columns:
            n_unique = df[col].nunique()
            dtype = str(df[col].dtype)
            stats_info[col] = f"Type: {dtype}, Unique: {n_unique}"
            if n_unique < 10:
                stats_info[col] += f", Examples: {list(df[col].unique())}"

        columns_summary = json.dumps(stats_info, ensure_ascii=False, indent=2)
        data_preview = df.head(3).to_markdown(index=False) # 縮減到3筆，夠了

        # === 優化 Prompt：強制極短標題 ===
        prompt = f"""
        你是一位精通 UX 設計的數據分析師。請分析以下資料，並提供 20 個最有價值的圖表建議。
        
        【欄位統計】：
        {columns_summary}
        
        【數據預覽】：
        {data_preview}
        
        【極致簡潔規則】：
        1. **標題極短**：標題嚴格限制在 **10 個中文字以內**。例如用「地區營收」取代「各地區的銷售金額總計」。
        2. **去除贅詞**：不要包含「分析」、「統計」、「圖表」等字眼。
        3. **多樣性**：包含趨勢、排行、佔比、交叉、分佈等不同視角。
        4. **雙軸防呆**：若建議雙軸圖，左右軸必須是不同欄位。
        5. **視覺防呆**：分類 > 20 不畫圓餅圖；分類 > 50 不畫長條圖(除非Top N)。
        
        請回傳 **純 JSON 格式**：
        [
            {{
                "group": "群組 (例如: 趨勢/排行/佔比/交叉)",
                "title": "極短標題 (Max 10字)",
                "chart_type": "圖表類型", 
                "x_col": "X軸欄位",
                "y_col": "Y軸欄位",
                "color_col": "顏色欄位(可null)",
                "sort": "desc/asc/none"
            }}
        ]
        
        【可用圖表類型】:
        "長條圖 (Bar)", "折線圖 (Line)", "雙軸組合圖 (Combo)", "圓餅圖 (Pie)", "樹狀圖 (TreeMap)", 
        "散佈圖 (Scatter)", "箱型圖 (Box Plot)", "面積圖 (Area)", "直方圖 (Histogram)", "雷達圖 (Radar)"
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
# 3. 使用說明書
# ==========================================
def get_manual_content():
    return """
# 📊 作圖小工具 (V37.1) 使用手冊

## 1. 🔑 啟動 AI
1. 前往 Google AI Studio 申請免費 Key。
2. 貼入左側「🔑 Gemini API Key」欄位並點擊驗證。

## 2. 🤖 智慧分析
上傳檔案後，AI 會自動生成 20 個精簡標題的分析建議。
點擊按鈕，圖表與設定會自動同步。

## 3. 🛠️ 常見問題
* **Q: 雙軸圖左右軸重複？**
  A: 程式已內建防呆，若重複會自動顯示為單軸，您可手動調整右軸為不同欄位。
* **Q: 圖表太密？**
  A: 使用左側「X 軸範圍」縮放，或切換為「趨勢圖」觀看。

祝您分析愉快！
    """

@st.cache_data
def generate_demo_excel():
    rows = 2000
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(rows)]
    locations = {'北區': ['台北', '新北'], '中區': ['台中', '新竹'], '南區': ['高雄', '台南']}
    regions, cities = [], []
    for _ in range(rows):
        r = random.choice(list(locations.keys()))
        regions.append(r)
        cities.append(random.choice(locations[r]))
    cats = ['電子', '家具', '家電']
    c_list = [random.choice(cats) for _ in range(rows)]
    df = pd.DataFrame({
        '訂單日期': dates, '地區': regions, '門市': cities, '產品類別': c_list,
        '銷售金額': np.random.randint(1000, 50000, rows),
        '利潤': np.random.randint(100, 5000, rows),
        '運送天數': np.random.poisson(3, rows),
        '滿意度': np.random.randint(1, 6, rows)
    })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='DemoData')
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
# 4. 主介面開始
# ==========================================
st.title("✨ 作圖小工具 (Gemini AI版)")

with st.sidebar:
    st.header("1. 資料來源")
    
    st.session_state['gemini_api_key'] = st.text_input("🔑 Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    
    if st.button("✅ 驗證 API Key", use_container_width=True):
        if not st.session_state['gemini_api_key']:
            st.error("請先輸入 Key")
        else:
            try:
                genai.configure(api_key=st.session_state['gemini_api_key'])
                models = list(genai.list_models())
                st.success(f"連線成功！")
            except Exception as e:
                st.error(f"連線失敗: {e}")

    st.markdown("---")
    
    with st.expander("📥 下載範例與說明書", expanded=False):
        st.download_button("📄 下載說明書", get_manual_content(), "User_Manual.txt")
        if st.button("🎲 生成範例資料"):
            excel_data = generate_demo_excel()
            st.download_button("📊 下載 Excel", excel_data, "Demo_Data.xlsx")
    
    font_choice = st.selectbox("字體", ["Microsoft JhengHei", "Arial"], index=0)
    inject_custom_css(font_choice)
    uploaded_files = st.file_uploader("上傳 Excel/CSV", type=["xlsx", "csv"], accept_multiple_files=True)

df = None
all_cols, num_cols, cat_cols = [], [], []

if uploaded_files:
    file_map = {f.name: f for f in uploaded_files}
    with st.sidebar: selected_file_name = st.selectbox("選擇檔案", list(file_map.keys()))
    current_file = file_map[selected_file_name]
    df = load_data(current_file)
    
    if df is not None:
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if date_cols:
            with st.sidebar:
                st.markdown("---")
                time_granularity = st.selectbox("⏳ 時間粒度", ["年-月 (Default)", "年", "季", "週", "日"])
            for col in date_cols:
                time_col_name = f"{col}(時間)"
                if time_granularity == "年": df[time_col_name] = df[col].dt.strftime('%Y')
                elif time_granularity == "季": df[time_col_name] = df[col].dt.to_period('Q').astype(str)
                elif time_granularity == "週": df[time_col_name] = df[col].dt.strftime('%Y-W%U')
                elif time_granularity == "日": df[time_col_name] = df[col].dt.strftime('%Y-%m-%d')
                else: df[time_col_name] = df[col].dt.strftime('%Y-%m')

        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        all_cols = df.columns.tolist()
        
        chart_types_list = ["長條圖 (Bar)", "折線圖 (Line)", "雙軸組合圖 (Combo)", "圓餅圖 (Pie)", "樹狀圖 (TreeMap)", "散佈圖 (Scatter)", "箱型圖 (Box Plot)", "面積圖 (Area)", "直方圖 (Histogram)", "雷達圖 (Radar)", "漏斗圖 (Funnel)"]
        agg_funcs_list = ["總和 (Sum)", "平均 (Avg)", "最大值 (Max)", "最小值 (Min)", "計數 (Count)"]
        sort_orders_list = ["預設 (依 X 軸)", "數值由大到小 (Desc)", "數值由小到大 (Asc)"]

        # === AI 分析建議區 ===
        st.markdown("---")
        st.subheader("🤖 Gemini 智慧分析建議")
        
        if st.session_state['gemini_api_key']:
            if 'last_analyzed_file' not in st.session_state or st.session_state['last_analyzed_file'] != selected_file_name:
                with st.spinner("🤖 AI 正在生成精簡分析建議..."):
                    insights, error_msg = analyze_with_gemini(df, st.session_state['gemini_api_key'])
                    if error_msg:
                        st.error(error_msg)
                    else:
                        st.session_state['ai_insights'] = insights
                        st.session_state['last_analyzed_file'] = selected_file_name
            
            if st.session_state.get('ai_insights'):
                insights = st.session_state['ai_insights']
                groups = sorted(list(set(ins['group'] for ins in insights)))
                
                for group_name in groups:
                    st.markdown(f"<div class='group-header'>{group_name}</div>", unsafe_allow_html=True)
                    cols = st.columns(3)
                    group_insights = [ins for ins in insights if ins['group'] == group_name]
                    for i, insight in enumerate(group_insights):
                        with cols[i % 3]:
                            if st.button(insight['title'], key=f"btn_{group_name}_{i}"):
                                c_type = insight.get('chart_type', '長條圖 (Bar)')
                                if c_type not in chart_types_list: c_type = '長條圖 (Bar)'
                                
                                st.session_state['chart_type_idx'] = chart_types_list.index(c_type)
                                st.session_state['chart_type_box'] = c_type
                                
                                sort_str = insight.get('sort', 'none')
                                s_idx = 0
                                if sort_str == 'desc': s_idx = 1
                                elif sort_str == 'asc': s_idx = 2
                                st.session_state['sort_order_idx'] = s_idx
                                st.session_state['sort_order_box'] = sort_orders_list[s_idx]

                                def sync_box(key_idx, key_box, col_list, target_val):
                                    if target_val and target_val in col_list:
                                        st.session_state[key_idx] = col_list.index(target_val)
                                        st.session_state[key_box] = target_val
                                    else:
                                        st.session_state[key_idx] = 0
                                        st.session_state[key_box] = col_list[0] if col_list else None

                                sync_box('x_col_idx', 'x_col_box', all_cols, insight.get('x_col'))
                                sync_box('y_col_idx', 'y_col_box', num_cols, insight.get('y_col'))
                                sync_box('y_col_2_idx', 'y_col_2_box', num_cols, insight.get('y_col')) 
                                sync_box('color_col_idx', 'color_col_box', ["(無)"]+all_cols, insight.get('color_col'))
                                
                                if c_type == "樹狀圖 (TreeMap)" and insight.get('x_col'):
                                    st.session_state['treemap_path'] = [insight.get('x_col')]
                                    st.session_state['treemap_box'] = [insight.get('x_col')]

                                st.rerun()
        else:
            st.warning("請輸入 API Key 以啟用智慧分析。")

        # === 側邊欄與繪圖設定 ===
        with st.sidebar:
            st.markdown("---")
            st.header("2. 繪圖設定")
            
            chart_type = st.selectbox("圖表類型", chart_types_list, index=st.session_state['chart_type_idx'], key='chart_type_box')
            
            x_col, y_col, y_col_2, color_col, facet_col = None, None, None, None, None
            agg_func = "sum"
            marker_symbol = "circle"
            agg_map = {"總和 (Sum)": "sum", "平均 (Avg)": "mean", "最大值 (Max)": "max", "最小值 (Min)": "min", "計數 (Count)": "count"}
            symbol_map_zh = {"圓形 (Circle)": "circle", "正方形 (Square)": "square", "菱形 (Diamond)": "diamond", "十字 (Cross)": "cross", "叉叉 (X)": "x", "三角形 (Triangle)": "triangle-up"}
            combo_types = ["長條圖 (Bar)", "折線圖 (Line)", "面積圖 (Area)", "散佈圖 (Scatter)"]

            if chart_type == "雙軸組合圖 (Combo)":
                st.info("💡 自由配：請分別設定左右軸")
                x_col = st.selectbox("X 軸 (共用)", all_cols, index=st.session_state['x_col_idx'], key='x_col_box')
                agg_label = st.selectbox("計算方式", agg_funcs_list, index=st.session_state['agg_func_idx'], key='agg_func_box')
                agg_func = agg_map[agg_label]
                st.markdown("---")
                col_L1, col_L2 = st.columns(2)
                with col_L1: type_L = st.selectbox("左軸類型", combo_types, index=0)
                with col_L2: y_col = st.selectbox("左軸數值", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                col_R1, col_R2 = st.columns(2)
                with col_R1: type_R = st.selectbox("右軸類型", combo_types, index=1)
                with col_R2: y_col_2 = st.selectbox("右軸數值", num_cols, index=st.session_state['y_col_2_idx'], key='y_col_2_box')
                color_col = st.selectbox("顏色分組", ["(無)"] + all_cols, index=st.session_state['color_col_idx'], key='color_col_box')
            elif chart_type in ["長條圖 (Bar)", "折線圖 (Line)", "面積圖 (Area)", "漏斗圖 (Funnel)", "雷達圖 (Radar)"]:
                x_col = st.selectbox("X 軸", all_cols, index=st.session_state['x_col_idx'], key='x_col_box')
                y_col = st.selectbox("Y 軸", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                agg_label = st.selectbox("計算方式", agg_funcs_list, index=st.session_state['agg_func_idx'], key='agg_func_box')
                agg_func = agg_map[agg_label]
                color_col = st.selectbox("顏色分組", ["(無)"] + all_cols, index=st.session_state['color_col_idx'], key='color_col_box')
                if chart_type not in ["雷達圖 (Radar)"]: facet_col = st.selectbox("拆分圖表 (Facet)", ["(無)"] + cat_cols, index=st.session_state['facet_col_idx'], key='facet_col_box')
                if chart_type == "折線圖 (Line)":
                    symbol_label = st.selectbox("點的形狀", list(symbol_map_zh.keys()))
                    marker_symbol = symbol_map_zh[symbol_label]
            elif chart_type == "樹狀圖 (TreeMap)":
                default_tree_path = st.session_state['treemap_path'] if st.session_state['treemap_path'] else (cat_cols[:2] if len(cat_cols)>=2 else cat_cols[:1])
                treemap_path = st.multiselect("層級結構", cat_cols, default=default_tree_path, key='treemap_box')
                y_col = st.selectbox("數值大小", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                color_col = st.selectbox("顏色依據", ["(無)"] + num_cols + cat_cols, index=st.session_state['color_col_idx'], key='color_col_box')
                agg_label = st.selectbox("計算方式", agg_funcs_list, index=st.session_state['agg_func_idx'], key='agg_func_box')
                agg_func = agg_map[agg_label]
            elif chart_type == "圓餅圖 (Pie)":
                x_col = st.selectbox("分類 (Label)", cat_cols, index=st.session_state.get('x_col_idx_pie', 0), key='x_col_box_pie')
                y_col = st.selectbox("數值 (Value)", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                agg_label = st.selectbox("計算方式", agg_funcs_list, index=st.session_state['agg_func_idx'], key='agg_func_box')
                agg_func = agg_map[agg_label]
            elif chart_type == "散佈圖 (Scatter)":
                x_col = st.selectbox("X 軸", all_cols, index=st.session_state['x_col_idx'], key='x_col_box')
                y_col = st.selectbox("Y 軸", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                color_col = st.selectbox("顏色分組", ["(無)"] + all_cols, index=st.session_state['color_col_idx'], key='color_col_box')
                symbol_label = st.selectbox("點的形狀", list(symbol_map_zh.keys()))
                marker_symbol = symbol_map_zh[symbol_label]
            elif chart_type == "箱型圖 (Box Plot)":
                x_col = st.selectbox("X 軸 (分組)", cat_cols, index=st.session_state.get('x_col_idx_box', 0), key='x_col_box_box')
                y_col = st.selectbox("Y 軸 (數值)", num_cols, index=st.session_state['y_col_idx'], key='y_col_box')
                color_col = st.selectbox("顏色分組", ["(無)"] + all_cols, index=st.session_state['color_col_idx'], key='color_col_box')
            elif chart_type == "直方圖 (Histogram)":
                x_col = st.selectbox("X 軸 (數值分佈)", num_cols, index=st.session_state.get('x_col_idx_hist', 0), key='x_col_box_hist')
                color_col = st.selectbox("顏色分組", ["(無)"] + cat_cols, index=st.session_state['color_col_idx'], key='color_col_box')

            st.markdown("---")
            st.header("3. 外觀與細節")
            user_x_min, user_x_max = None, None
            if x_col and x_col in df.columns:
                with st.expander("🔎 縮放 X 軸範圍 (X-Axis Range)", expanded=False):
                    if pd.api.types.is_datetime64_any_dtype(df[x_col]) or "訂單日期" in x_col or "date" in x_col.lower():
                        try:
                            min_date = df[x_col].min().date() if hasattr(df[x_col].min(), 'date') else df[x_col].min()
                            max_date = df[x_col].max().date() if hasattr(df[x_col].max(), 'date') else df[x_col].max()
                            c1, c2 = st.columns(2)
                            with c1: user_x_min = st.date_input("開始日期", min_date)
                            with c2: user_x_max = st.date_input("結束日期", max_date)
                        except: st.caption("無法自動偵測日期範圍")
                    elif pd.api.types.is_numeric_dtype(df[x_col]):
                        min_val = float(df[x_col].min())
                        max_val = float(df[x_col].max())
                        c1, c2 = st.columns(2)
                        with c1: user_x_min = st.number_input("最小值", value=min_val)
                        with c2: user_x_max = st.number_input("最大值", value=max_val)
                    else: st.caption("此 X 軸為文字類別，不支援範圍縮放。")

            if chart_type in ["長條圖 (Bar)", "折線圖 (Line)", "雙軸組合圖 (Combo)", "面積圖 (Area)"]:
                with st.expander("📊 排序與參考線 (Sorting & Ref Line)", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1: sort_order = st.selectbox("排序方式", sort_orders_list, index=st.session_state['sort_order_idx'], key='sort_order_box')
                    with c2: ref_line_val = st.number_input("添加目標/參考線", value=st.session_state['ref_line_val'], step=1000.0, key='ref_line_box')
            else: sort_order, ref_line_val = "預設 (依 X 軸)", 0.0

            with st.expander("🔠 字體與軸設定", expanded=False):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    legend_font_size = st.slider("圖例文字大小", 10, 30, 14)
                    legend_pos = st.selectbox("圖例位置", ["右側 (預設)", "上方", "下方", "隱藏"])
                with col_f2:
                    x_axis_font_size = st.slider("X 軸文字大小", 10, 30, 14)
                    y_axis_font_size = st.slider("Y 軸文字大小", 10, 30, 14)

            if chart_type in ["折線圖 (Line)", "散佈圖 (Scatter)"] or (chart_type == "雙軸組合圖 (Combo)" and ("折線圖" in type_L or "散佈圖" in type_L or "折線圖" in type_R or "散佈圖" in type_R)):
                with st.expander("🔷 點的樣式 (Marker)", expanded=False): marker_size = st.slider("點的大小", 4, 25, 10)
            else: marker_size = 10 

            with st.expander("🔢 數值標籤 (Data Labels)", expanded=False):
                show_label = st.checkbox("顯示圖中數字", value=True)
                if show_label:
                    c1, c2 = st.columns(2)
                    with c1: decimal_places = st.selectbox("小數位數", [0, 1, 2, 3, 4], index=0)
                    with c2: label_size = st.slider("數字大小", 10, 30, 16)
                    label_position = st.selectbox("數字位置", ["上方 (Top)", "下方 (Bottom)", "置中 (Middle)", "自動 (Auto)"], index=0)
                    text_format = f'.{decimal_places}f'
                else: text_format, label_size, label_position = None, 14, "top center"

        if df is not None:
            legend_config = dict(font=dict(size=legend_font_size))
            if legend_pos == "右側 (預設)": legend_config.update(dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
            elif legend_pos == "上方": legend_config.update(dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
            elif legend_pos == "下方": legend_config.update(dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
            elif legend_pos == "隱藏": legend_config.update(dict(visible=False))

            pos_map_line = {"上方 (Top)": "top center", "下方 (Bottom)": "bottom center", "置中 (Middle)": "middle center", "自動 (Auto)": "top center"}
            pos_map_bar = {"上方 (Top)": "outside", "下方 (Bottom)": "inside", "置中 (Middle)": "inside", "自動 (Auto)": "outside"}
            plotly_pos_line = pos_map_line.get(label_position, "top center")
            plotly_pos_bar = pos_map_bar.get(label_position, "outside")

            layout_config = dict(font=dict(family=font_choice, size=18, color="#333"), title_font=dict(size=24), legend=legend_config, template="plotly_white", height=600, margin=dict(t=80, b=80, l=80, r=50))

            def create_trace(sub_df, x_c, y_c, c_type, axis_name, color_code, show_txt, txt_fmt, m_size, opac=1.0):
                common_args = dict(x=sub_df[x_c], y=sub_df[y_c], name=f"{y_c} ({axis_name})", text=sub_df[y_c] if show_txt else None, texttemplate=f'%{{text:{txt_fmt}}}' if show_txt else None, textfont=dict(size=label_size))
                if "長條圖" in c_type: return go.Bar(**common_args, marker_color=color_code, opacity=opac, textposition=plotly_pos_bar, cliponaxis=False) 
                elif "折線圖" in c_type:
                    common_args['mode'] = 'lines+markers+text' if show_txt else 'lines+markers'
                    trace = go.Scatter(**common_args, line=dict(width=3, color=color_code), marker=dict(size=m_size, symbol=marker_symbol))
                    if show_txt: trace.update(textposition=plotly_pos_line) 
                    return trace
                elif "面積圖" in c_type: return go.Scatter(**common_args, fill='tozeroy', line=dict(width=2, color=color_code), mode='lines')
                elif "散佈圖" in c_type:
                    common_args['mode'] = 'markers+text' if show_txt else 'markers'
                    trace = go.Scatter(**common_args, marker=dict(size=m_size, color=color_code, symbol=marker_symbol))
                    if show_txt: trace.update(textposition=plotly_pos_line)
                    return trace
                return go.Bar(**common_args)

            try:
                fig = None
                if chart_type == "雙軸組合圖 (Combo)":
                    grp_cols = [x_col]
                    if color_col != "(無)" and color_col in df.columns: grp_cols.append(color_col)
                    
                    metrics = [y_col]
                    if y_col_2 != y_col: metrics.append(y_col_2)
                        
                    df_agg = df.groupby(grp_cols, as_index=False)[metrics].agg(agg_func)
                    
                    if sort_order == "數值由大到小 (Desc)": df_agg = df_agg.sort_values(by=y_col, ascending=False)
                    elif sort_order == "數值由小到大 (Asc)": df_agg = df_agg.sort_values(by=y_col, ascending=True)

                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    if color_col != "(無)" and color_col in df.columns and "長條圖" in type_L:
                        for c in df_agg[color_col].unique():
                            subset = df_agg[df_agg[color_col] == c]
                            fig.add_trace(go.Bar(x=subset[x_col], y=subset[y_col], name=f"{c}-{y_col}", text=subset[y_col] if show_label else None, texttemplate=f'%{{text:{text_format}}}' if show_label else None, textfont=dict(size=label_size), opacity=0.7, textposition=plotly_pos_bar, cliponaxis=False), secondary_y=False)
                    else:
                        trace_L = create_trace(df_agg, x_col, y_col, type_L, "左軸", '#636EFA', show_label, text_format, marker_size, 0.7)
                        fig.add_trace(trace_L, secondary_y=False)
                    
                    if y_col_2 in df_agg.columns:
                        trace_R = create_trace(df_agg, x_col, y_col_2, type_R, "右軸", '#EF553B', show_label, text_format, marker_size, 0.9)
                        fig.add_trace(trace_R, secondary_y=True)
                        
                    fig.update_layout(title=f"自由雙軸分析: {y_col} vs {y_col_2}")

                elif chart_type in ["長條圖 (Bar)", "折線圖 (Line)", "面積圖 (Area)", "漏斗圖 (Funnel)", "雷達圖 (Radar)"]:
                    grp_cols = [x_col]
                    if color_col != "(無)" and color_col in df.columns: grp_cols.append(color_col)
                    if facet_col and facet_col != "(無)" and facet_col in df.columns: grp_cols.append(facet_col)
                    df_agg = df.groupby(grp_cols, as_index=False)[y_col].agg(agg_func)
                    if sort_order == "數值由大到小 (Desc)": df_agg = df_agg.sort_values(by=y_col, ascending=False)
                    elif sort_order == "數值由小到大 (Asc)": df_agg = df_agg.sort_values(by=y_col, ascending=True)
                    
                    params = {"data_frame": df_agg, "x": x_col, "y": y_col, "title": f"{y_col} 分析"}
                    if color_col != "(無)" and color_col in df.columns: params["color"] = color_col
                    if facet_col and facet_col != "(無)" and facet_col in df.columns: params["facet_col"] = facet_col
                    if show_label: params["text"] = y_col
                    if "長條圖" in chart_type:
                        fig = px.bar(**params)
                        if show_label: fig.update_traces(texttemplate=f'%{{text:{text_format}}}', textposition=plotly_pos_bar, textfont_size=label_size, cliponaxis=False)
                    elif "折線圖" in chart_type:
                        fig = px.line(**params, markers=True)
                        fig.update_traces(marker_symbol=marker_symbol, marker_size=marker_size, line_width=3)
                        if show_label: fig.update_traces(texttemplate=f'%{{text:{text_format}}}', textposition=plotly_pos_line, textfont_size=label_size)
                    elif "面積圖" in chart_type: fig = px.area(**params)
                    elif "漏斗圖" in chart_type:
                        fig = px.funnel(**params)
                        if show_label: fig.update_traces(textinfo="value", texttemplate=f'%{{value:{text_format}}}', textfont_size=label_size)
                    elif "雷達圖" in chart_type:
                        fig = px.line_polar(df_agg, r=y_col, theta=x_col, line_close=True, color=color_col if color_col != "(無)" and color_col in df.columns else None, title=params['title'])
                        fig.update_traces(fill='toself')

                elif chart_type == "樹狀圖 (TreeMap)":
                    if not treemap_path: st.warning("請至少選擇一個層級欄位")
                    else:
                        df_agg = df.groupby(treemap_path, as_index=False)[y_col].agg(agg_func)
                        fig = px.treemap(df, path=treemap_path, values=y_col, color=color_col if color_col != "(無)" and color_col in df.columns else y_col, title=f"層級分析: {' > '.join(treemap_path)}")
                        fig.update_traces(textinfo="label+value+percent entry")
                elif chart_type == "箱型圖 (Box Plot)":
                    params = {"data_frame": df, "x": x_col, "y": y_col, "title": f"{y_col} 分佈情形"}
                    if color_col != "(無)" and color_col in df.columns: params["color"] = color_col
                    fig = px.box(**params)
                elif chart_type == "圓餅圖 (Pie)":
                    df_agg = df.groupby(x_col, as_index=False)[y_col].agg(agg_func)
                    fig = px.pie(df_agg, values=y_col, names=x_col, title=f"{x_col} 佔比")
                    info_mode = 'percent+label'
                    if show_label: info_mode += '+value'
                    fig.update_traces(textposition='inside', textinfo=info_mode, textfont_size=label_size)
                elif chart_type == "散佈圖 (Scatter)":
                    params = {"data_frame": df, "x": x_col, "y": y_col, "title": f"{x_col} vs {y_col}"}
                    if color_col != "(無)" and color_col in df.columns: params["color"] = color_col
                    fig = px.scatter(**params)
                    fig.update_traces(marker_symbol=marker_symbol, marker_size=marker_size)
                    if show_label: fig.update_traces(text=df[y_col], textposition=plotly_pos_line, textfont_size=label_size)
                elif chart_type == "直方圖 (Histogram)":
                    params = {"data_frame": df, "x": x_col, "title": f"{x_col} 分佈"}
                    if color_col != "(無)" and color_col in df.columns: params["color"] = color_col
                    fig = px.histogram(**params)

                if fig:
                    fig.update_layout(**layout_config)
                    fig.update_xaxes(tickfont=dict(size=x_axis_font_size), title_font=dict(size=x_axis_font_size+4))
                    fig.update_yaxes(tickfont=dict(size=y_axis_font_size), title_font=dict(size=y_axis_font_size+4))
                    if user_x_min is not None and user_x_max is not None: fig.update_xaxes(range=[user_x_min, user_x_max])
                    if chart_type != "圓餅圖 (Pie)" and chart_type != "樹狀圖 (TreeMap)" and ref_line_val > 0:
                        fig.add_hline(y=ref_line_val, line_dash="dash", line_color="red", annotation_text=f"Target: {ref_line_val}", annotation_position="top left", annotation_font=dict(size=14, color="red"))
                    if sort_order != "預設 (依 X 軸)": fig.update_xaxes(type='category')
                    
                    config = {'toImageButtonOptions': {'format': 'png', 'filename': '4k_chart_export', 'height': 1080, 'width': 1920, 'scale': 2 }, 'displayModeBar': True }
                    st.plotly_chart(fig, use_container_width=True, config=config)
                    
                    col_d1, col_d2 = st.columns([1, 4])
                    with col_d1:
                        buffer = io.StringIO()
                        fig.write_html(buffer, include_plotlyjs='cdn')
                        html_bytes = buffer.getvalue().encode()
                        st.download_button("📥 下載互動式 HTML", data=html_bytes, file_name="chart.html", mime="text/html")
                    with col_d2:
                            st.success("✨ **分析完成**：點擊上方的「戰略分析建議」可快速切換不同視角。")
                    with st.expander("查看數據表"):
                        if 'df_agg' in locals(): st.dataframe(df_agg, use_container_width=True)
                        else: st.dataframe(df.head(100), use_container_width=True)
            except Exception as e:
                st.error(f"發生錯誤: {e}")
else:
    st.info("👋 歡迎使用作圖小工具，請上傳 Excel 開始。")