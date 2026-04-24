import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import io

# 【視覚設定】Matplotlibでグラフ内の日本語を文字化けさせないための設定
plt.rcParams['font.family'] = 'MS Gothic' 

# ==========================================
# 1. ページ基本設定
# ==========================================
st.set_page_config(page_title="ジョイントカプラ工法の断面解析", layout="wide")# 画面レイアウトを「wide（横広）」に設定。
# ==========================================
# 2. 高速化のためのキャッシュ設定 (@st.cache_data)
# ==========================================
@st.cache_data
def calculate_nm_short_term_cached(xn, D_out, D_in, fca, Ec, fsa, Es, rebar_coords_json, a_s_each):
    """
    短期許容限界（弾性解析）の計算def
    xn（中立軸）を仮定し、断面内の歪み分布から積分によって N と M を逆算します。
    """
    rebar_coords = pd.read_json(io.StringIO(rebar_coords_json))# 【データ変換】JSON文字列で受け取った座標を、計算可能な表形式に
    R = D_out / 2
    y_rebar = rebar_coords["y (mm)"].values # NumPy配列として取り出す
    y_rebar_min = y_rebar.min()   # 最外縁アンカーまでの有効高さ
    d_rebar_max = R - y_rebar_min # 最外縁アンカーまでの有効高さ
    # 【力学条件】コンクリートの縁応力度、または引張側アンカーの応力度が許容値に達する時の曲率を算出。
    phi_c = (fca / Ec) / xn
    phi_s = (fsa / Es) / (d_rebar_max - xn) if (d_rebar_max - xn) > 0 else float('inf')
    phi = min(phi_c, phi_s) # 安全側に小さい方の曲率を採用
    # 【数値積分】断面を120層にスライスして、各層の応力×面積を足し合わせます（シンプソン法に近い簡易積分）。
    num_div = 80 
    y_scan = np.linspace(max(-R, R - xn), R, num_div)
    dy = y_scan[1] - y_scan[0] if len(y_scan) > 1 else 0
    Nc_c, Mc_c = 0.0, 0.0
    # 120層分の計算を一気に行う（配列同士の計算）
    dist_all = R - y_scan
    sig_c_all = phi * Ec * (xn - dist_all)
    # NumPyのnp.sqrtを使い、マイナス値を0でクリップ
    widths = 2 * np.sqrt(np.maximum(0, R**2 - y_scan**2))
    dA_all = widths * dy
    Nc_c = np.sum(sig_c_all * dA_all)      # 全層の軸力を一気に合計
    Mc_c = np.sum(sig_c_all * dA_all * y_scan) # 全層の曲げを一気に合計
    # 【個別計算】アンカー1本ずつの座標における応力度を計算。
    Ns, Ms = 0.0, 0.0
    # --- 鉄筋計算のベクトル化 ---
    dist_s = R - y_rebar
    sig_s_all = phi * Es * (xn - dist_s)
    # 条件分岐(if)もnp.whereで一括処理
    sig_c_at_rebar = np.where(dist_s <= xn, phi * Ec * (xn - dist_s), 0)
    forces = (sig_s_all - sig_c_at_rebar) * a_s_each
    Ns = np.sum(forces)
    Ms = np.sum(forces * y_rebar)
    return Nc_c + Ns, Mc_c + Ms

@st.cache_data
def calculate_ultimate_nm_cached(theta, D, Db, Fc, Fb, ab, nb, cru, kappa):
    """
    終局強度（塑性解析）の計算エンジン
    設計指針の数式に基づき、等価応力ブロックと仮想鋼管厚を用いて解析的に解きます。
    """
    t = (nb * ab) / (np.pi * Db) # アンカー本数を円周上に均した「仮想の鋼管厚さ」に置換
    # 資料数式 (3.27)〜(3.30) の実装
    Ncu = cru * (D**2) * Fc * (theta - np.sin(theta) * np.cos(theta)) / 4
    Nbu = Db * Fb * (2 * theta - np.pi) * t * kappa
    Mcu = cru * (D**3) * Fc * (np.sin(theta)**3) / 12
    Mbu = (Db**2) * Fb * t * np.sin(theta) * kappa
    return (Ncu + Nbu) / 1000.0, (Mcu + Mbu) / 1e6 # 単位を kN, kNm に変換
# ==========================================
# 3. サイドバー：入力エリア
# ==========================================
# ここで定義された変数は、変更されるたびにアプリ全体を再実行させる「トリガー」になります。
with st.sidebar:
    st.title("計算諸条件")
    st.header("1.断面寸法")
    D_pile = st.number_input("杭径 Dp (mm)", value=800, step=100, min_value=500, max_value=1200)
    D_outer = D_pile + 200 * 2 # 施工誤差や被りを考慮した「仮想径 D」
    st.header("2. コンクリート")
    Fc = st.number_input("設計基準強度 Fc (N/mm2)", value=30, step=3, min_value=21, max_value=60)
    # 【自動計算】設計基準強度に基づき、ヤング係数や弾性係数比を自動判定。
    gamma_fac = 23 if Fc <= 36 else 23.5
    raw_Ec = 3.35 * 10**4 * ((gamma_fac / 24) * 1)**2 * (Fc / 60)**(1/3)
    Ec = round(raw_Ec / 100) * 100 # 実務慣習に合わせ100単位で丸め
    if   Fc <= 27: Ne = 15
    elif Fc <= 36: Ne = 13
    elif Fc < 48:  Ne = 11
    else :         Ne = 9
    # 【設計制限】杭径に対して配置可能なアンカーの最大本数を制限。
    # 施工上の鉄筋間隔を確保するための独自の安全フラグです。
    rebar_limit_dict = {400:9, 500:11, 600:13, 700:16, 800:18, 900:20, 1000:22, 1100:24, 1200:26}
    rebar_limit = rebar_limit_dict.get(D_pile, 26)
    st.header("3. 異形棒鋼")
    Fs = 490.0 # SD490を想定
    Es = Ec * Ne
    rebar_area_each = 1340 # D41の標準断面積
    rebar_count = st.number_input("本数", value=6, min_value=4, max_value=rebar_limit, step=2)
    rebar_d_circle = D_pile + 70 # アンカー配置円直径 Db
    # 許容応力度の設定（短期：Fcの2/3、終局係数：0.85）
    fs_short, fc_short, cru, kappa = Fs, Fc*(2/3), 0.85, 1.0
    st.write("---")
    rebar_mode = st.radio("配置方法の選択", ["CSVから取得", "本数で均等割り"])
# ==========================================
# 4. 鉄筋データの管理（変更検知ロジック）
# ==========================================
# 「ユーザーが手動で打ち換えたデータ」を保護しつつ、
# 「杭径などを変えたら自動でリセットする」ための高度な管理エリアです。
csv_file = "rebar_patterns.csv"
# 【初期化】初回実行時に、現在のパラメータを記憶する。
if 'last_params' not in st.session_state:
    st.session_state.last_params = {"D": D_pile, "nb": rebar_count, "mode": rebar_mode}
# 【変更検知】サイドバーの重要項目が前回と違うか比較。
params_changed = (
    st.session_state.last_params["D"] != D_pile or 
    st.session_state.last_params["nb"] != rebar_count or 
    st.session_state.last_params["mode"] != rebar_mode
)
# パラメータが変わった時、またはデータが存在しない時だけ、座標データを再生成または読み込み。
# これにより、単なる「再計算ボタン」クリックで手動修正データが消えるのを防ぎます。
if params_changed or 'rebar_data' not in st.session_state:
    if rebar_mode == "CSVから取得" and os.path.exists(csv_file):
        all_patterns = pd.read_csv(csv_file)
        filtered = all_patterns[(all_patterns['D_pile'] == D_pile) & (all_patterns['nb'] == rebar_count)]
        if not filtered.empty:
            st.session_state.rebar_data = filtered[['x', 'y']].rename(columns={'x':'x (mm)','y':'y (mm)'}).reset_index(drop=True)
        else:
            st.sidebar.warning("一致する配置位置がなし。均等配置で作成生。")
            angles = np.linspace(0, 2 * np.pi, int(rebar_count), endpoint=False)
            st.session_state.rebar_data = pd.DataFrame({
                "x (mm)": np.round((rebar_d_circle / 2) * np.sin(angles + np.pi/2), 1),
                "y (mm)": np.round((rebar_d_circle / 2) * np.cos(angles + np.pi/2), 1)
            })
    else:
        # 【均等配置生成】円周を本数分で等分割した $x, y$ 座標を生成。
        angles = np.linspace(0, 2 * np.pi, int(rebar_count), endpoint=False)
        st.session_state.rebar_data = pd.DataFrame({
            "x (mm)": np.round((rebar_d_circle / 2) * np.sin(angles + np.pi/2), 1),
            "y (mm)": np.round((rebar_d_circle / 2) * np.cos(angles + np.pi/2), 1)
        })
    # 変更された状態を「前回の状態」として上書き保存。
    st.session_state.last_params = {"D": D_pile, "nb": rebar_count, "mode": rebar_mode}

# ==========================================
# 5. 計算のループ実行
# ==========================================
# 準備された座標データを用いて、実際に曲線を描くための点群を算出。
current_rebar_df = st.session_state.rebar_data
rebar_json = current_rebar_df.to_json() # キャッシュ判定用の「指紋」としてJSON化
nb_current = len(current_rebar_df)
area_s_all = nb_current * rebar_area_each
area_c_all = np.pi * (D_outer / 2)**2

# --- A. 短期許容限界の計算 ---
short_results = []
# 端点1：純引張（全アンカーが許容引張に達した状態）
n_pure_tension_s = -(area_s_all * fs_short) / 1000.0
short_results.append([n_pure_tension_s, 0.0])

# 中間点：中立軸 xn を変化させて計算。xn=x_bal（つりあい状態）を必ず含める。
eps_ca, eps_sa = fc_short/Ec, fs_short/Es
d_rebar_max = (D_outer/2) - current_rebar_df["y (mm)"].min()
x_bal = (eps_ca / (eps_ca + eps_sa)) * d_rebar_max
xn_list = np.sort(np.append(np.logspace(0, 4, 97), x_bal))

for x in xn_list:
    n_raw, m_raw = calculate_nm_short_term_cached(x, D_outer, 0, fc_short, Ec, fs_short, Es, rebar_json, rebar_area_each)
    short_results.append([n_raw / 1000.0, m_raw / 1e6])

# 端点2：純圧縮（全断面が圧縮許容に達した状態）
fs_prime = min(fs_short, (fc_short / Ec) * Es)
n_full_comp_s = (fc_short * (area_c_all - area_s_all) + fs_prime * area_s_all) / 1000.0
short_results.append([n_full_comp_s, 0.0])
df_short = pd.DataFrame(short_results, columns=["N", "M"]).sort_values("N")

# --- B. 終局強度の計算 ---
# 角度 $\theta$ を0（純引張）から $\pi$（純圧縮）まで100段階で変化させる。
ult_results = []
for th in np.linspace(0, np.pi, 100):
    nu, mu = calculate_ultimate_nm_cached(th, D_outer, rebar_d_circle, Fc, Fs, rebar_area_each, nb_current, cru, kappa)
    ult_results.append([nu, mu])
df_ult = pd.DataFrame(ult_results, columns=["N", "M"]).sort_values("N")

# ==========================================
# 6. 解析結果の表示
# ==========================================
st.header("断面解析結果")
col_plot, col_info = st.columns([4, 2])
with col_plot:
    st.subheader("N-M 相関図")
    fig, ax = plt.subplots(figsize=(6, 3.71))#黄金比
    # グラフの重なりを考慮し、塗りつぶし(alpha)と実線を組み合わせて描画。
    ax.plot(df_short["N"], df_short["M"], color='blue', lw=2, label="Short term")
    ax.fill_between(df_short["N"], df_short["M"], 0, color='blue', alpha=0.05)
    ax.plot(df_ult["N"], df_ult["M"], color='red', lw=2, label="Ultimate")
    ax.fill_between(df_ult["N"], df_ult["M"], 0, color='red', alpha=0.05)
    # 【表示調整】耐力曲線が中心に来るよう、X軸（軸力）とY軸（曲げ）の範囲を動的に計算。
    n_min_all, n_max_all = min(df_short["N"].min(), df_ult["N"].min()), max(df_short["N"].max(), df_ult["N"].max())
    m_max_all = max(df_short["M"].max(), df_ult["M"].max())
    ax.set_xlim(n_min_all * 1.3, n_max_all * 1.3); ax.set_ylim(0, m_max_all * 1.2) # Y軸は必ず0から開始。
    ax.set_xlabel("N (kN)"); ax.set_ylabel("M (kNm)")
    ax.grid(True, linestyle=':', alpha=0.5)
    # X軸、Y軸のゼロラインを強調。
    ax.axhline(0, color='black', lw=1); ax.axvline(0, color='black', lw=1)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # --- CSVエクスポート機能 ---
    # 設計者のエビデンス資料として使えるよう、短期と終局を並べたCSVを作成。
    csv_export_df = pd.concat([
        df_short.rename(columns={"N": "Short_N(kN)", "M": "Short_M(kNm)"}).reset_index(drop=True),
        df_ult.rename(columns={"N": "Ult_N(kN)", "M": "Ult_M(kNm)"}).reset_index(drop=True)
        ], axis=1)
    
    st.download_button(
        label="CSV保存",
        data=csv_export_df.to_csv(index=False).encode('utf-8-sig'),
        file_name=f"NM_Results_Dp{D_pile}_Fc{Fc}_n{rebar_count}.csv",
        mime="text/csv",
    )

with col_info:
    st.subheader("概略断面図")
    fig_sec, ax_sec = plt.subplots(figsize=(4, 4))
    st.write("")   # 普通の改行
    st.write("")   # 普通の改行
    r_outer = D_outer / 2
    # 円や点を「add_artist」や「scatter」で重ねて、断面の構成を視覚化。
    ax_sec.add_artist(plt.Circle((0,0), r_outer, color='#e0e0e0', ec='black' ,linestyle='--'))  # 仮想径円
    ax_sec.add_artist(plt.Circle((0,0), D_pile/2, fill=False, ec='black', linestyle='-'))         # 杭径円
    ax_sec.scatter(current_rebar_df["x (mm)"], current_rebar_df["y (mm)"], color='black', s=41)   # アンカー位置
    ax_sec.axis('off')# XY軸の数値は不要
    ax_sec.axhline(0, color='black', lw=0.8, linestyle='-.', alpha=0.5) # X軸（水平線）
    ax_sec.axvline(0, color='black', lw=0.8, linestyle='-.', alpha=0.5) # Y軸（垂直線）

    # 杭径に関わらず一定のカメラアングルで見えるよう表示限界を固定。
    limit = (1200 + 400)/2 * 1.05 
    ax_sec.set_xlim(-limit, limit); ax_sec.set_ylim(-limit, limit)
    ax_sec.set_aspect('equal')
    fig_sec.tight_layout()
    st.pyplot(fig_sec)

# ==========================================
# 7. 鉄筋配置の編集
# ==========================================
st.write("---")
st.subheader("異形棒鋼の配置位置の編集")
st.info("※表の数値を書き換え後、「再計算」ボタンを押すとグラフが更新されます。")
# st.form を使うことで、数値を1つ変えるたびに画面がリロードされるのを防ぎ、
# 編集がすべて終わった段階で一括計算させます（パフォーマンス対策）。
with st.form("rebar_edit_form"):
    # 本数が多い場合に備え、横長（転置）の表形式で表示。
    transposed_df = st.session_state.rebar_data.T
    edited_transposed_df = st.data_editor(
        transposed_df,
        use_container_width=True,
        key="rebar_editor"
    )
    
    col_submit, _ = st.columns([1, 4])
    with col_submit:
        # type="primary" はボタンを青く強調し、メインアクションであることを示します。
        submit_button = st.form_submit_button("再計算", type="primary")

if submit_button:
    # 編集完了後、データを元の縦長形式に戻して保存し、強制リロードをかけて計算に反映させます。
    st.session_state.rebar_data = edited_transposed_df.T
    st.rerun()
