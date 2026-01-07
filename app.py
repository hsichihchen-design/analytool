import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import numpy as np
import random
import re
from datetime import datetime, timedelta
import google.generativeai as genai
import json

# ==========================================
# 0. 初始化 Session State
# ==========================================
default_states = {
    'menu_id': 0, 
    'chart_type_idx': 0, 
    'x_col_idx': 0,
    'y_col_idx': 0,
    'y_col_2_idx': 0,
    'color_col_idx': 0,
    'agg_func_idx': 0,
    'sort_order_idx': 0,
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
st.set_page_config(page_title="作圖小工具 V84 (Insight Maximize)", layout="wide", page_icon="✨")

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
        .block-container {{ padding-top: 1rem !important; padding-bottom: 2rem !important; }}
        header {{ visibility: hidden; }}
        div.stButton > button {{
            width: 100%; min-height: 40px; height: 100%; white-space: normal; word-wrap: break-word;
            padding: 2px 6px; line-height: 1.1; border-radius: 4px; border: 1px solid #ddd;
            background-color: #ffffff; text-align: left; display: flex; align-items: center;
            font-size: 0.8rem; box-shadow: 0 1px 1px rgba(0,0,0,0.05);
            font-family: {font_css_rule} !important;
        }}
        div.stButton > button:hover {{
            border-color: #7c4dff; color: #7c4dff; background-color: #f8f5ff;
            transform: translateY(-1px); z-index: 1;
        }}
        .group-header {{
            font-weight: 700; font-size: 0.85rem; color: #666;
            margin-top: 10px; margin-bottom: 5px; padding-bottom: 2px;
            border-bottom: 1px solid #eee;
            font-family: {font_css_rule} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 核心功能：Gemini AI 分析引擎 (V84 多樣性增強版)
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

        # [Step 1] 資料剖析
        stats_info = {}
        num_df = df.select_dtypes(include=['number'])
        corr_hints = []
        if len(num_df.columns) > 1:
            try:
                corr_matrix = num_df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                strong_pairs = upper.stack().reset_index()
                strong_pairs.columns = ['col1', 'col2', 'corr']
                strong_pairs = strong_pairs[strong_pairs['corr'] > 0.5].sort_values(by='corr', ascending=False).head(5)
                for _, row in strong_pairs.iterrows():
                    corr_hints.append(f"{row['col1']} & {row['col2']} (Corr: {row['corr']:.2f})")
            except: pass

        for col in df.columns:
            n_unique = df[col].nunique()
            dtype = str(df[col].dtype)
            col_profile = {
                "dtype": dtype, "n_unique": n_unique,
                "missing_pct": round(df[col].isnull().mean() * 100, 1)
            }
            if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 0:
                col_profile["min"] = float(df[col].min())
                col_profile["max"] = float(df[col].max())
                col_profile["mean"] = float(df[col].mean())
                if (col_profile["max"] > 190000 and col_profile["max"] < 210012):
                    col_profile["semantic_hint"] = "可能為年月格式 (YYYYMM)"
            else:
                try:
                    vc = df[col].value_counts().head(5)
                    top_counts = {str(k): int(v) for k, v in vc.items()} 
                    col_profile["top_frequent_values"] = top_counts
                except: pass

            try: col_profile["samples"] = df[col].dropna().astype(str).sample(min(3, len(df))).tolist()
            except: col_profile["samples"] = []
            stats_info[col] = col_profile

        data_summary = {
            "columns_profile": stats_info,
            "correlation_hints (High Priority)": corr_hints,
            "total_rows": len(df)
        }
        
        columns_summary_json = json.dumps(data_summary, ensure_ascii=False, indent=2)

        # [Step 2] V84 Advanced Prompt
        prompt = f"""
        <role>
        你是一位極具洞察力的資深數據科學家。你的目標是挖掘數據中「不易被發現」的規律，並透過多樣化的圖表呈現。
        **請避免懶惰地只生成長條圖和折線圖，請積極思考是否有更適合的高階圖表。**
        </role>

        <data_profile>
        {columns_summary_json}
        </data_profile>

        <chart_catalog>
        1. "長條圖 (Bar)": 基礎比較。
        2. "折線圖 (Line)": 趨勢分析。
        3. "面積圖 (Area)": 累積趨勢。
        4. "圓餅圖 (Pie)": 簡單佔比。
        5. "雙軸組合圖 (Combo)": **強烈推薦** (關聯性/對比)。
        6. "散佈圖 (Scatter)": **強烈推薦** (分佈/關聯)。
        7. "箱型圖 (Box Plot)": **強烈推薦** (離群值/穩定性)。
        8. "直方圖 (Histogram)": 頻率分佈。
        9. "漏斗圖 (Funnel)": 階段轉化。
        10. "樹狀圖 (TreeMap)": **強烈推薦** (複雜層級/大量類別佔比)。
        11. "雷達圖 (Radar)": **強烈推薦** (多維指標評估)。
        </chart_catalog>

        <scenario_triggers>
        **當你發現以下特徵時，請優先使用對應圖表：**
        * **多個評分/比率欄位 (Score/Rate/%)** -> 務必生成 1~2 個 **"雷達圖 (Radar)"** 來比較綜合表現。
        * **大量分類 (Category > 5) 且有層級關係** -> 務必生成 **"樹狀圖 (TreeMap)"** (比圓餅圖更好)。
        * **數值分佈/檢測異常值** -> 務必生成 **"箱型圖 (Box Plot)"** (例如：利潤分佈、運送天數分佈)。
        * **兩個數值變數** -> 優先考慮 **"散佈圖 (Scatter)"** 或 **"雙軸組合圖 (Combo)"**。
        </scenario_triggers>

        <diversity_rules>
        1. 生成 20 個建議。
        2. **約束：** "長條圖" 與 "折線圖" 的總數 **不得超過 10 個** (<= 50%)。
        3. **必須包含：** 至少 1 個箱型圖、1 個樹狀圖、1 個雙軸圖 (如果資料允許)。
        </diversity_rules>

        <output_format>
        Strict JSON Array only:
        [
          {{
            "group": "分析視角 (例如: 異常偵測, 綜合評估)",
            "title": "標題 (Max 15字)",
            "chart_type": "Chart Catalog 中的標準名稱",
            "x_col": "欄位名",
            "y_col": "數值欄位名",
            "color_col": "分組欄位 (若無則 null)",
            "sort": "desc/asc/none"
          }}
        ]
        </output_format>
        """

        response = model.generate_content(prompt)
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if match:
            json_str = match.group(0)
            insights = json.loads(json_str)
            return insights, None
        else:
            return None, "AI 回傳格式無法解析 (No JSON found)"

    except Exception as e:
        return None, f"AI 分析失敗: {str(e)}"

# [Helper] 模糊搜尋
def find_best_match(target, candidates):
    if not target: return None
    if target in candidates: return target
    for c in candidates:
        if target in c or c in target: return c
    return None

# ==========================================
# 3. 輔助與資料載入
# ==========================================

@st.cache_data
def generate_demo_excel():
    rows = 800
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(rows)]
    ym_int = [int(d.strftime('%Y%m')) for d in dates]
    regions = ['北區', '中區', '南區']
    cities_map = {'北區': ['台北市', '新北市', '基隆市'], '中區': ['台中市', '新竹市', '苗栗縣'], '南區': ['高雄市', '台南市', '嘉義市']}
    row_regions, row_cities = [], []
    for _ in range(rows):
        r = random.choice(regions)
        row_regions.append(r)
        row_cities.append(cities_map[r][random.randint(0, len(cities_map[r])-1)])
    stages = ['1_瀏覽商品', '2_加入購物車', '3_結帳流程', '4_完成訂單']
    row_stages = random.choices(stages, weights=[0.4, 0.3, 0.2, 0.1], k=rows)
    products = ['旗艦機 Pro', '輕旗艦 Air', '入門機 SE', '電競機 GT']
    row_products = [random.choice(products) for _ in range(rows)]
    prices = np.random.randint(5000, 40000, rows)
    units = np.random.randint(1, 10, rows)
    sales = prices * units
    margins = np.random.uniform(0.1, 0.4, rows)
    profit = sales * margins
    df = pd.DataFrame({
        '訂單日期': dates, '年月份': ym_int, '大區域': row_regions, '城市': row_cities,
        '產品型號': row_products, '銷售階段': row_stages, '訂單金額': sales, '訂單利潤': profit,
        '毛利率': margins, '折扣率': np.random.choice([0, 0.05, 0.1, 0.2], rows),
        '運送天數': np.random.randint(1, 7, rows),
        '效能評分': np.random.randint(6, 10, rows), '外觀評分': np.random.randint(5, 10, rows),
        'CP值評分': np.random.randint(4, 10, rows), '售後評分': np.random.randint(6, 10, rows),
        '續航評分': np.random.randint(5, 10, rows)
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

with st.sidebar:
    st.markdown("### ✨ Lyra V84")
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
    with st.expander("📥 超級測試資料"):
        if st.button("🎲 生成測試資料"):
            st.download_button("📊 下載 Excel", generate_demo_excel(), "Lyra_Full_Test_Data.xlsx")

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
        # 時間粒度增強
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        for col in date_cols:
            df[f"{col}(YM)"] = df[col].dt.strftime('%Y-%m')

        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        all_cols = df.columns.tolist()
        agg_funcs_list = ["總和 (Sum)", "平均 (Avg)", "最大值 (Max)", "計數 (Count)"]

        # --- 繪圖設定區 (Sidebar) ---
        with st.sidebar:
            st.markdown("---")
            st.header("2. 繪圖設定")
            
            uid = st.session_state['menu_id']

            def update_idx(key_name, options_list):
                val = st.session_state.get(f"{key_name}_idx", 0)
                if val >= len(options_list): val = 0
                return val

            # 1. 圖表類型
            chart_type_idx = update_idx('chart_type', CHART_TYPES)
            chart_type = st.selectbox("圖表類型", CHART_TYPES, index=chart_type_idx, key=f"chart_type_{uid}")
            if chart_type in CHART_TYPES:
                st.session_state['chart_type_idx'] = CHART_TYPES.index(chart_type)

            # 初始化繪圖變數
            x_col, y_col, y_col_2, color_col = None, None, None, "(無)"
            agg_func = "總和 (Sum)"
            treemap_path = []

            # UI 條件渲染
            if chart_type == "雙軸組合圖 (Combo)":
                x_col = st.selectbox("X 軸", all_cols, index=update_idx('x_col', all_cols), key=f'x_col_{uid}')
                y_col = st.selectbox("左軸數值", num_cols, index=update_idx('y_col', num_cols), key=f'y_col_{uid}')
                y_col_2 = st.selectbox("右軸數值", num_cols, index=update_idx('y_col_2', num_cols), key=f'y_col_2_{uid}')
                agg_func = st.selectbox("計算", agg_funcs_list, index=update_idx('agg_func', agg_funcs_list), key=f'agg_func_{uid}')
                color_col = "(無)" 
            elif chart_type == "樹狀圖 (TreeMap)":
                raw_default = st.session_state['treemap_path'] if st.session_state['treemap_path'] else (cat_cols[:2] if len(cat_cols)>=2 else cat_cols[:1])                
                valid_defaults = [c for c in raw_default if c in cat_cols]                
                treemap_path = st.multiselect("層級結構", cat_cols, default=valid_defaults, key=f'treemap_{uid}')            
                y_col = st.selectbox("數值大小", num_cols, index=update_idx('y_col', num_cols), key=f'y_col_{uid}')
                color_col = st.selectbox("顏色依據", ["(無)"] + num_cols + cat_cols, index=update_idx('color_col', ["(無)"]+num_cols+cat_cols), key=f'color_col_{uid}')
                agg_func = st.selectbox("計算", agg_funcs_list, index=update_idx('agg_func', agg_funcs_list), key=f'agg_func_{uid}')
            elif chart_type == "雷達圖 (Radar)":
                x_col = st.selectbox("維度 (Label)", cat_cols, index=update_idx('x_col', cat_cols), key=f'x_col_{uid}')
                y_col = st.selectbox("數值 (Value)", num_cols, index=update_idx('y_col', num_cols), key=f'y_col_{uid}')
                color_col = st.selectbox("分組", ["(無)"] + all_cols, index=update_idx('color_col', ["(無)"]+all_cols), key=f'color_col_{uid}')
                agg_func = st.selectbox("計算", agg_funcs_list, index=update_idx('agg_func', agg_funcs_list), key=f'agg_func_{uid}')
            else:
                x_col = st.selectbox("X 軸", all_cols, index=update_idx('x_col', all_cols), key=f'x_col_{uid}')
                y_col = st.selectbox("Y 軸 (數值)", num_cols, index=update_idx('y_col', num_cols), key=f'y_col_{uid}')
                agg_func = st.selectbox("計算", agg_funcs_list, index=update_idx('agg_func', agg_funcs_list), key=f'agg_func_{uid}')
                color_col = st.selectbox("分組", ["(無)"] + all_cols, index=update_idx('color_col', ["(無)"]+all_cols), key=f'color_col_{uid}')
            
            # 手動同步
            if x_col and x_col in all_cols: st.session_state['x_col_idx'] = all_cols.index(x_col)
            if y_col and y_col in num_cols: st.session_state['y_col_idx'] = num_cols.index(y_col)
            if y_col_2 and y_col_2 in num_cols: st.session_state['y_col_2_idx'] = num_cols.index(y_col_2)
            if color_col in (["(無)"] + all_cols): st.session_state['color_col_idx'] = (["(無)"] + all_cols).index(color_col)
            if agg_func in agg_funcs_list: st.session_state['agg_func_idx'] = agg_funcs_list.index(agg_func)
            if treemap_path: st.session_state['treemap_path'] = treemap_path

        # ==========================================
        # 5. 優先執行繪圖引擎
        # ==========================================
        current_chart_fig = None
        df_agg = None
        
        try:
            agg_map = {"總和 (Sum)": "sum", "平均 (Avg)": "mean", "最大值 (Max)": "max", "計數 (Count)": "count"}
            real_agg = agg_map[agg_func]
            use_raw_data = chart_type in RAW_DATA_CHARTS
            
            if use_raw_data:
                df_agg = df.copy()
            else:
                if chart_type == "雙軸組合圖 (Combo)":
                     grp_cols = [x_col]
                     measure_cols = list(set([y_col, y_col_2])) 
                     df_agg = df.groupby(grp_cols, as_index=False)[measure_cols].agg(real_agg)
                elif chart_type == "樹狀圖 (TreeMap)":
                     if not treemap_path: df_agg = None
                     else:
                         agg_dict = {y_col: real_agg}
                         if color_col != "(無)" and color_col != y_col and color_col in num_cols:
                             agg_dict[color_col] = real_agg
                         df_agg = df.groupby(treemap_path, as_index=False).agg(agg_dict)
                elif chart_type == "雷達圖 (Radar)":
                     grp_cols = [x_col]
                     if color_col != "(無)": grp_cols.append(color_col)
                     df_agg = df.groupby(grp_cols, as_index=False)[y_col].agg(real_agg)
                else:
                    grp_cols = [x_col]
                    if color_col != "(無)": grp_cols.append(color_col)
                    df_agg = df.groupby(grp_cols, as_index=False)[y_col].agg(real_agg)

            # 後處理：數值轉字串
            if df_agg is not None and x_col and x_col in df_agg.columns and pd.api.types.is_numeric_dtype(df_agg[x_col]):
                col_mean = df_agg[x_col].mean()
                if (1900 < col_mean < 2100) or (190000 < col_mean < 210012):
                    df_agg[x_col] = df_agg[x_col].astype(str)

            # 強制排序
            if chart_type in ["折線圖 (Line)", "面積圖 (Area)", "雙軸組合圖 (Combo)"] and df_agg is not None and x_col:
                df_agg = df_agg.sort_values(by=x_col, ascending=True)
            elif not use_raw_data and df_agg is not None and chart_type not in ["樹狀圖 (TreeMap)", "雷達圖 (Radar)"]:
                sort_idx = st.session_state['sort_order_idx']
                if sort_idx == 1: df_agg = df_agg.sort_values(by=y_col, ascending=False)
                elif sort_idx == 2: df_agg = df_agg.sort_values(by=y_col, ascending=True)
            
            # 繪圖
            if df_agg is not None:
                common_params = {"data_frame": df_agg, "x": x_col if (x_col and x_col in df_agg.columns) else None, "title": f"{chart_type}: {x_col if x_col else ''}"}
                if color_col != "(無)" and color_col in df_agg.columns: common_params["color"] = color_col

                if chart_type == "長條圖 (Bar)":
                    current_chart_fig = px.bar(**common_params, y=y_col, text_auto='.2s')
                elif chart_type == "折線圖 (Line)":
                    current_chart_fig = px.line(**common_params, y=y_col, markers=True)
                elif chart_type == "面積圖 (Area)":
                    current_chart_fig = px.area(**common_params, y=y_col)
                elif chart_type == "漏斗圖 (Funnel)":
                    current_chart_fig = px.funnel(**common_params, y=y_col)
                elif chart_type == "雙軸組合圖 (Combo)":
                    current_chart_fig = make_subplots(specs=[[{"secondary_y": True}]])
                    current_chart_fig.add_trace(go.Bar(x=df_agg[x_col], y=df_agg[y_col], name=y_col, marker_color='#636EFA', opacity=0.8), secondary_y=False)
                    current_chart_fig.add_trace(go.Scatter(x=df_agg[x_col], y=df_agg[y_col_2], name=y_col_2, mode='lines+markers', line=dict(color='#EF553B', width=3)), secondary_y=True)
                    current_chart_fig.update_layout(title=f"{y_col} vs {y_col_2}")
                elif chart_type == "圓餅圖 (Pie)":
                    current_chart_fig = px.pie(df_agg, values=y_col, names=x_col, title=f"{x_col} 佔比")
                elif chart_type == "樹狀圖 (TreeMap)":
                    current_chart_fig = px.treemap(df_agg, path=treemap_path, values=y_col, color=color_col if color_col!="(無)" else y_col, title="層級分析")
                elif chart_type == "雷達圖 (Radar)":
                     current_chart_fig = px.line_polar(df_agg, r=y_col, theta=x_col, line_close=True, color=color_col if color_col != "(無)" else None, title="雷達分析")
                     current_chart_fig.update_traces(fill='toself')
                elif chart_type == "直方圖 (Histogram)":
                    current_chart_fig = px.histogram(df_agg, x=x_col, color=color_col if color_col!="(無)" else None, title=f"{x_col} 分佈")
                elif chart_type == "箱型圖 (Box Plot)":
                    current_chart_fig = px.box(df_agg, x=x_col, y=y_col, color=color_col if color_col!="(無)" else None, title=f"{y_col} 分佈 (by {x_col})")
                elif chart_type == "散佈圖 (Scatter)":
                    current_chart_fig = px.scatter(df_agg, x=x_col, y=y_col, color=color_col if color_col!="(無)" else None, title=f"{x_col} vs {y_col}")

        except Exception as e:
            st.error(f"繪圖錯誤: {e}")

        # -------------------------------------------------------
        # 顯示圖表
        # -------------------------------------------------------
        if current_chart_fig:
            current_chart_fig.update_layout(
                template="plotly_white", 
                height=450, 
                margin=dict(t=30, b=10),
                font=dict(family=font_choice.split(',')[0].strip("'"), size=16), 
                hovermode="x unified"
            )
            st.plotly_chart(current_chart_fig, use_container_width=True)
        else:
            st.info("👈 請從左側選擇圖表類型，或等待下方 AI 產生建議。")

        # ==========================================
        # 6. AI 分析與控制面板
        # ==========================================
        
        if st.session_state['gemini_api_key']:
            if 'last_analyzed_file' not in st.session_state or st.session_state['last_analyzed_file'] != selected_file_name:
                with st.spinner("🧠 正在構建分析矩陣 (全面掃描 11 種圖表可能性)..."):
                    insights, error_msg = analyze_with_gemini(df, st.session_state['gemini_api_key'])
                    if not error_msg:
                        st.session_state['ai_insights'] = insights
                        st.session_state['last_analyzed_file'] = selected_file_name
                    else:
                        st.error(error_msg)

            if st.session_state.get('ai_insights'):
                insights = st.session_state['ai_insights']
                groups = sorted(list(set(ins['group'] for ins in insights)))
                
                st.markdown(f"**🤖 AI 深度分析建議** (共 {len(insights)} 項)")
                with st.container(height=200, border=True):
                    
                    for group_name in groups:
                        st.markdown(f"<div class='group-header'>{group_name}</div>", unsafe_allow_html=True)
                        cols = st.columns(5)
                        group_insights = [ins for ins in insights if ins['group'] == group_name]
                        for i, insight in enumerate(group_insights):
                            with cols[i % 5]:
                                if st.button(insight['title'], key=f"btn_{group_name}_{i}"):
                                    raw_type = insight.get('chart_type', '')
                                    matched_type = "長條圖 (Bar)"
                                    for standard_type in CHART_TYPES:
                                        keywords = standard_type.split(' ')
                                        if any(k.strip("()") in raw_type for k in keywords if len(k)>2):
                                            matched_type = standard_type
                                            break
                                    
                                    try: st.session_state['chart_type_idx'] = CHART_TYPES.index(matched_type)
                                    except: st.session_state['chart_type_idx'] = 0

                                    def sync(key, ai_val, candidates):
                                        best_val = find_best_match(ai_val, candidates)
                                        if best_val: 
                                            st.session_state[f"{key}_idx"] = candidates.index(best_val)
                                        else: 
                                            st.session_state[f"{key}_idx"] = 0
                                    
                                    sync('x_col', insight.get('x_col'), all_cols)
                                    sync('y_col', insight.get('y_col'), num_cols)
                                    sync('y_col_2', insight.get('y_col'), num_cols) 
                                    sync('color_col', insight.get('color_col'), ["(無)"]+all_cols)
                                    
                                    sort_str = insight.get('sort', 'none')
                                    if sort_str == 'desc': st.session_state['sort_order_idx'] = 1
                                    elif sort_str == 'asc': st.session_state['sort_order_idx'] = 2
                                    else: st.session_state['sort_order_idx'] = 0
                                    
                                    if matched_type == "樹狀圖 (TreeMap)":
                                        target_x = find_best_match(insight.get('x_col'), cat_cols)
                                        if target_x:
                                            st.session_state['treemap_path'] = [target_x]
                                            if '城市' in all_cols and target_x == '大區域':
                                                st.session_state['treemap_path'] = ['大區域', '城市']
                                    
                                    st.session_state['menu_id'] += 1
                                    st.rerun()
