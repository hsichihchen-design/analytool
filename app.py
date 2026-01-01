import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import numpy as np
import random
from datetime import datetime, timedelta
import re 

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
    'treemap_path': []
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==========================================
# 1. 全域設定與 CSS
# ==========================================
st.set_page_config(page_title="作圖小工具 V31", layout="wide", page_icon="📊")

def inject_custom_css(font_family):
    st.markdown(f"""
    <style>
        html, body, [class*="css"] {{
            font-family: '{font_family}', 'Microsoft JhengHei', sans-serif !important;
        }}
        .stDownloadButton button {{ width: 100%; border-color: #4CAF50; color: #4CAF50; }}
        /* 智慧建議按鈕樣式 */
        div.stButton > button {{
            width: 100%; min-height: 60px; height: auto; white-space: normal; word-wrap: break-word;
            padding: 10px 15px; line-height: 1.5; border-radius: 8px; border: 1px solid #e0e0e0;
            background-color: #ffffff; text-align: left; display: flex; align-items: center;
        }}
        div.stButton > button:hover {{
            border-color: #FF4B4B; color: #FF4B4B; background-color: #fffbfb;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); transform: translateY(-2px);
        }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 核心功能：具備語意理解的計分引擎 (Smart Scoring Engine)
# ==========================================

def get_column_score(col_name, data_series, role):
    """
    根據欄位名稱與資料特性，計算該欄位適合某個角色(Role)的分數。
    Role: 'metric' (數值), 'dimension' (分類), 'date' (日期)
    """
    score = 0
    col_str = str(col_name).lower()
    
    # === A. 關鍵字加權 (Keyword Heuristics) ===
    keywords = {
        'metric': ['amount', 'sales', 'profit', 'cost', 'price', 'qty', 'quantity', 'revenue', 'margin', 'score', 
                   '金額', '銷售', '營收', '利潤', '毛利', '成本', '數量', '單價', '分數', '人次'],
        'dimension': ['region', 'city', 'country', 'category', 'type', 'status', 'segment', 'brand', 'source', 'manager', 'rep',
                      '地區', '城市', '國家', '類別', '型態', '狀態', '分群', '品牌', '來源', '業務', '經理', '部門'],
        'date': ['date', 'time', 'year', 'month', 'day', 'quarter', 'week', 
                 '日期', '時間', '年', '月', '日', '季', '週']
    }
    
    id_keywords = ['id', 'no', 'code', 'phone', 'zip', 'lat', 'lon', 'year', 'month', 'day', # year有時不適合作為加總數值
                   '編號', '代碼', '電話', '郵遞', '經度', '緯度']

    if any(k in col_str for k in keywords[role]):
        score += 10 
    
    if role == 'metric':
        if any(k in col_str for k in id_keywords):
            score -= 20 

    # === B. 統計特徵加權 (Statistical Heuristics) ===
    n_unique = data_series.nunique()
    
    if role == 'dimension':
        if 1 < n_unique < 50: score += 5 
        if n_unique > 100: score -= 10   
        if n_unique == 1: score -= 5      
        if data_series.dtype == 'object': score += 2 

    if role == 'metric':
        if pd.api.types.is_numeric_dtype(data_series):
            score += 5
            # 排除看起來像年份的數字
            if data_series.mean() > 1900 and data_series.mean() < 2100 and data_series.std() < 5:
                score -= 10 
        else:
            score -= 100 

    return score

def generate_insights_advanced(df):
    insights = []
    
    # 1. 智慧盤點欄位 (Smart Column Picking)
    all_num_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
    
    # --- 找出最佳日期欄位 ---
    time_cols = [c for c in df.columns if '(時間)' in c] # 優先用系統生成的粒度欄位
    if time_cols:
        best_date_col = time_cols[0] 
    else:
        raw_date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        best_date_col = raw_date_cols[0] if raw_date_cols else None

    # --- 找出最佳數值欄位 (Top 3 Metrics) ---
    metric_scores = []
    for col in all_num_cols:
        s = get_column_score(col, df[col], 'metric')
        metric_scores.append((col, s))
    metric_scores.sort(key=lambda x: x[1], reverse=True)
    top_metrics = [m[0] for m in metric_scores if m[1] > 0][:3]
    
    # --- 找出最佳分類欄位 (Top 3 Dimensions) ---
    dim_scores = []
    for col in all_cat_cols:
        s = get_column_score(col, df[col], 'dimension')
        dim_scores.append((col, s))
    dim_scores.sort(key=lambda x: x[1], reverse=True)
    top_dims = [d[0] for d in dim_scores if d[1] > 0][:3]

    if not top_metrics: return []

    chart_types = ["長條圖 (Bar)", "折線圖 (Line)", "雙軸組合圖 (Combo)", "圓餅圖 (Pie)", "樹狀圖 (TreeMap)", 
                   "散佈圖 (Scatter)", "箱型圖 (Box Plot)", "面積圖 (Area)", "直方圖 (Histogram)", "雷達圖 (Radar)"]
    sort_orders = ["預設 (依 X 軸)", "數值由大到小 (Desc)", "數值由小到大 (Asc)"]

    # === 生成建議策略 (Strategies) ===

    # 策略 A: 趨勢 (Trend)
    if best_date_col:
        for num in top_metrics[:2]:
            insights.append({
                "group": "📈 趨勢分析 (Trend)",
                "title": f"{num} 的時間走勢",
                "params": {"chart_type_idx": chart_types.index("折線圖 (Line)"), "x_col_name": best_date_col, "y_col_name": num, "agg_func_idx": 0, "sort_order_idx": 0}
            })

    # 策略 B: 排行 (Ranking)
    for cat in top_dims[:2]: 
        num = top_metrics[0]
        insights.append({
            "group": "🏆 重點排行 (Ranking)",
            "title": f"各 {cat} 的 {num} 表現",
            "params": {"chart_type_idx": chart_types.index("長條圖 (Bar)"), "x_col_name": cat, "y_col_name": num, "agg_func_idx": 0, "sort_order_idx": sort_orders.index("數值由大到小 (Desc)")}
        })

    # 策略 C: 交叉 (Cross)
    if len(top_dims) >= 2:
        c1, c2 = top_dims[0], top_dims[1]
        num = top_metrics[0]
        insights.append({
            "group": "📊 交叉分析 (Cross)",
            "title": f"{c1} 與 {c2} 的分佈",
            "params": {"chart_type_idx": chart_types.index("長條圖 (Bar)"), "x_col_name": c1, "y_col_name": num, "color_col_name": c2, "agg_func_idx": 0, "sort_order_idx": sort_orders.index("數值由大到小 (Desc)")}
        })

    # 策略 D: 結構 (Composition)
    for cat in top_dims:
        n = df[cat].nunique()
        num = top_metrics[0]
        if n <= 5:
            insights.append({
                "group": "🍰 結構佔比 (Share)",
                "title": f"{cat} 的 {num} 佔比",
                "params": {"chart_type_idx": chart_types.index("圓餅圖 (Pie)"), "x_col_name": cat, "y_col_name": num, "agg_func_idx": 0, "sort_order_idx": 0}
            })
        elif 5 < n < 20:
             insights.append({
                "group": "🍰 結構佔比 (Share)",
                "title": f"{cat} 的規模 (樹狀圖)",
                "params": {"chart_type_idx": chart_types.index("樹狀圖 (TreeMap)"), "treemap_path": [cat], "y_col_name": num, "agg_func_idx": 0, "sort_order_idx": 0}
            })
             break 

    # 策略 E: 關聯 (Correlation)
    if len(top_metrics) >= 2:
        n1, n2 = top_metrics[0], top_metrics[1]
        insights.append({
            "group": "🔗 相關性 (Correlation)",
            "title": f"{n1} vs {n2} 關聯",
            "params": {"chart_type_idx": chart_types.index("散佈圖 (Scatter)"), "x_col_name": n1, "y_col_name": n2, "agg_func_idx": 0, "sort_order_idx": 0}
        })

    return insights

# ==========================================
# 3. 資料生成與說明書 (V31版)
# ==========================================
def get_manual_content():
    return """
# 📊 作圖小工具 (V31) 使用手冊

## 1. 📂 資料準備 (關鍵第一步)
本工具內建「語意分析 AI」，為了讓它精準判讀，請準備 **「一維明細表 (流水帳)」**。

### ✅ 正確格式範例
每一列 (Row) 是一筆獨立紀錄，第一列是標題。
| 訂單日期 | 地區 | 產品名稱 | 銷售金額 | 利潤 |
| :--- | :--- | :--- | :--- | :--- |
| 2024-01-01 | 台北 | 智慧手機 | 25000 | 5000 |
| 2024-01-02 | 台中 | 無線耳機 | 3000 | 800 |

### ❌ 常見錯誤 (請避免)
1. **統計報表**：不要上傳已經算好「1月總計、2月總計」的表格。
2. **合併儲存格**：請取消所有合併，確保程式能讀取每一格。
3. **特殊符號**：金額欄位請用純數字 (如 `1000`)，不要包含 `$`、`NTD` 或 `元`。

## 2. 🤖 智慧分析操作 (Strategic Insights)
上傳檔案後，系統會自動執行以下動作：
1. **語意偵測**：自動尋找關鍵字 (如：Sales, Amount, Date, Region...)。
2. **生成建議**：在主畫面產生分類好的分析按鈕：
   - 📈 **趨勢 (Trend)**：自動繪製時間走勢圖。
   - 🏆 **排行 (Ranking)**：自動列出前幾名的分類排行。
   - 📊 **交叉 (Cross)**：分析兩個維度 (如地區 vs 產品) 的分佈。
   - 🔗 **關聯 (Correlation)**：若有多個數值，自動分析相關性。
3. **一鍵生成**：點擊按鈕，圖表與左側設定會自動跳轉到位！

## 3. 🛠️ 手動微調與進階設定
您依然可以在左側側邊欄進行細微調整：
- **⏳ 時間粒度**：若有日期欄位，可一鍵切換 年/季/月/週/日 視角。
- **📊 圖表類型**：支援 雙軸圖 (Combo)、樹狀圖 (TreeMap)、雷達圖等 11 種圖表。
- **🔢 美化設定**：
  - **排序**：設定「數值由大到小」讓長條圖更整齊。
  - **目標線**：輸入 KPI 數字，圖表會顯示紅色虛線。
  - **數值標籤**：可設定顯示位數與位置 (如置中、上方)。

## 4. 💾 輸出成果
- **📷 下載圖片**：滑鼠移至圖表右上角的相機圖示，下載 4K 高畫質 PNG。
- **📥 下載 HTML**：點擊下方綠色按鈕，下載可互動的網頁檔。

祝您分析順利！
    """

@st.cache_data
def generate_demo_excel():
    rows = 20000
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 730)) for _ in range(rows)]
    locations = {'北區': ['台北信義', '新北板橋'], '中區': ['台中旗艦', '新竹巨城'], '南區': ['高雄巨蛋', '台南西門']}
    regions, cities = [], []
    for _ in range(rows):
        r = random.choice(list(locations.keys()))
        regions.append(r)
        cities.append(random.choice(locations[r]))
    cats = ['消費電子', '辦公家具', '生活家電']
    prods = {'消費電子': ['手機', '耳機'], '辦公家具': ['工學椅', '升降桌'], '生活家電': ['清淨機', '氣炸鍋']}
    c_list, p_list = [], []
    for _ in range(rows):
        c = random.choice(cats)
        c_list.append(c)
        p_list.append(random.choice(prods[c]))
    df = pd.DataFrame({
        '訂單日期': dates, '地區': regions, '門市': cities, '產品類別': c_list, '產品名稱': p_list,
        '銷售渠道': np.random.choice(['官網', '門市', '蝦皮'], rows),
        '銷售階段': np.random.choice(['1.接觸', '2.詢價', '3.下單', '4.結案'], rows, p=[0.4, 0.3, 0.2, 0.1]),
        '業務員': np.random.choice(['小明', '大華', '美美', '志豪'], rows),
        '銷售金額': np.random.randint(1000, 50000, rows),
        '利潤': np.random.randint(-500, 10000, rows),
        '運送天數': np.random.poisson(3, rows) + 1,
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
st.title("📊 作圖小工具")

with st.sidebar:
    st.header("1. 資料來源")
    with st.expander("📥 下載範例與說明書 (Resources)", expanded=True):
        manual_txt = get_manual_content()
        st.download_button(label="📄 下載使用說明書 (.txt)", data=manual_txt, file_name="User_Manual.txt", mime="text/plain")
        if st.button("🎲 生成並下載 2萬筆範例資料"):
            with st.spinner("正在生成..."):
                excel_data = generate_demo_excel()
                st.download_button(label="📊 點此儲存範例 Excel (.xlsx)", data=excel_data, file_name="Demo_Big_Data_20k.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("---")
    font_choice = st.selectbox("字體", ["Microsoft JhengHei", "華康粗圓體", "華康儷中黑", "Arial"], index=0)
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
        # 時間粒度
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if date_cols:
            with st.sidebar:
                st.markdown("---")
                st.header("⏳ 時間設定")
                time_granularity = st.selectbox("時間粒度", ["年-月 (Default)", "年 (Year)", "季 (Quarter)", "週 (Week)", "日 (Day)"])
            for col in date_cols:
                time_col_name = f"{col}(時間)"
                if time_granularity == "年 (Year)": df[time_col_name] = df[col].dt.strftime('%Y')
                elif time_granularity == "季 (Quarter)": df[time_col_name] = df[col].dt.to_period('Q').astype(str)
                elif time_granularity == "週 (Week)": df[time_col_name] = df[col].dt.strftime('%Y-W%U')
                elif time_granularity == "日 (Day)": df[time_col_name] = df[col].dt.strftime('%Y-%m-%d')
                else: df[time_col_name] = df[col].dt.strftime('%Y-%m')

        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        all_cols = df.columns.tolist()

        # === 戰略分析建議區 ===
        st.markdown("---")
        st.subheader("💡 戰略分析建議 (Strategic Insights)")
        
        insights = generate_insights_advanced(df)
        
        if not insights:
            st.info("⚠️ 偵測不到足夠的關鍵欄位。請確認資料是否有「數值欄」(如金額) 與「分類欄」(如地區)。")
        else:
            groups = sorted(list(set(ins['group'] for ins in insights)))
            for group_name in groups:
                group_insights = [ins for ins in insights if ins['group'] == group_name]
                st.markdown(f"**{group_name}**")
                cols = st.columns(3)
                for i, insight in enumerate(group_insights):
                    with cols[i % 3]:
                        if st.button(insight['title'], key=f"btn_{group_name}_{i}"):
                            params = insight["params"]
                            st.session_state['chart_type_idx'] = params["chart_type_idx"]
                            st.session_state['agg_func_idx'] = params["agg_func_idx"]
                            st.session_state['sort_order_idx'] = params["sort_order_idx"]
                            
                            def get_idx(lst, name): 
                                try: return lst.index(name)
                                except: return 0
                            
                            if "x_col_name" in params: st.session_state['x_col_idx'] = get_idx(all_cols, params["x_col_name"])
                            if "y_col_name" in params: st.session_state['y_col_idx'] = get_idx(num_cols, params["y_col_name"])
                            if "color_col_name" in params: st.session_state['color_col_idx'] = get_idx(["(無)"]+all_cols, params["color_col_name"])
                            else: st.session_state['color_col_idx'] = 0
                            if "treemap_path" in params: st.session_state['treemap_path'] = params["treemap_path"]
                            
                            st.rerun()
                st.markdown("---")

        # === 側邊欄與繪圖設定 ===
        with st.sidebar:
            st.markdown("---")
            st.header("2. 繪圖設定")
            chart_types_list = ["長條圖 (Bar)", "折線圖 (Line)", "雙軸組合圖 (Combo)", "圓餅圖 (Pie)", "樹狀圖 (TreeMap)", "散佈圖 (Scatter)", "箱型圖 (Box Plot)", "面積圖 (Area)", "直方圖 (Histogram)", "雷達圖 (Radar)", "漏斗圖 (Funnel)"]
            agg_funcs_list = ["總和 (Sum)", "平均 (Avg)", "最大值 (Max)", "最小值 (Min)", "計數 (Count)"]
            sort_orders_list = ["預設 (依 X 軸)", "數值由大到小 (Desc)", "數值由小到大 (Asc)"]
            
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
                    df_agg = df.groupby(grp_cols, as_index=False)[[y_col, y_col_2]].agg(agg_func)
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