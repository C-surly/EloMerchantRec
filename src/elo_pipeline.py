# -*- coding: utf-8 -*-
"""
Elo Merchant Category Recommendation — 优化版完整建模 Pipeline
================================================================
对照通用 Baseline 的全流程深度优化(内存压缩 / 分层清洗 / 双表分层聚合 /
时序与交叉特征 / 特征筛选 / 分层10折 / Optuna 贝叶斯调参 / LGB+XGB+CAT /
Outlier 二分类 / Ridge Stacking / 加权融合),兼容本地与 Kaggle Notebook。

用法:
    python elo_pipeline.py                 # 全量运行(特征生成→训练→融合→保存)
    CONFIG 中 DEBUG=True 可抽样快速验证;TUNE=True 开启 Optuna 调参

依赖:pandas / numpy / scikit-learn / lightgbm / xgboost / catboost / optuna /
      pyarrow / matplotlib(均为主流库,Kaggle 镜像自带)
"""

import gc
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无显示环境下出图
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ============================================================================
# 0. 全局配置(本地 / Kaggle 自适应)
# ============================================================================
_KAGGLE_DIR = "/kaggle/input/elo-merchant-category-recommendation"
CONFIG = {
    # 路径:Kaggle Notebook 中自动切换到官方数据集挂载目录
    "DATA_DIR": _KAGGLE_DIR if os.path.exists(_KAGGLE_DIR) else "data/raw",
    "PROC_DIR": "/kaggle/working" if os.path.exists(_KAGGLE_DIR) else "data/processed",
    "OUT_DIR": "/kaggle/working" if os.path.exists(_KAGGLE_DIR) else "outputs",
    # 复现性(ELO_SEED 环境变量可覆盖,用于多 seed 平均)
    "SEED": int(os.environ.get("ELO_SEED", 2019)),
    "N_FOLDS": 10,          # 任务书要求:分层 10 折
    "N_THREADS": 16,  # 20万样本规模下 16 线程最优;更多线程同步开销反而拖慢
    # 参考日:new 表最晚交易为 2018-04-30,统一以次日为时间锚点
    "REF_DATE": "2018-05-01",
    # 特征筛选保留维数(方差过滤 + 重要性筛选后)
    "TOP_K": 300,
    # DEBUG:抽样 card_id 端到端快速验证(折数/轮数同步降低)
    "DEBUG": False,
    "DEBUG_CARDS": 20000,
    # Optuna 贝叶斯调参开关(默认关,使用下方预调优参数保证开箱复现)
    "TUNE": False,
    "TUNE_TRIALS": 40,
    # 可选:额外训练"无 outlier 干净模型"用于 top-N 后处理消融
    "TRAIN_CLEAN_MODEL": False,
    "OUTLIER_TARGET": -33.21928095,
}

t0 = time.time()


def log(msg: str):
    """带耗时戳的进度日志。"""
    print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)


# ============================================================================
# 1. 数据预处理层
# ============================================================================
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """全表内存压缩:数值类型向下转换(int64→int8/16/32,float64→float32)。

    提分逻辑:29M 行交易表不压缩极易 OOM;压缩后同样内存可容纳更多中间特征。
    """
    start = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        t = df[col].dtype
        if pd.api.types.is_integer_dtype(t):
            mn, mx = df[col].min(), df[col].max()
            for cand in (np.int8, np.int16, np.int32):
                if np.iinfo(cand).min <= mn and mx <= np.iinfo(cand).max:
                    df[col] = df[col].astype(cand)
                    break
        elif pd.api.types.is_float_dtype(t):
            df[col] = df[col].astype(np.float32)
    if verbose:
        end = df.memory_usage().sum() / 1024 ** 2
        log(f"  内存压缩 {start:.1f}MB -> {end:.1f}MB ({100 * (start - end) / start:.0f}% 减少)")
    return df


TRANS_DTYPES = {  # 读取即指定 dtype,避免 64 位默认类型的读入峰值
    "card_id": "object", "city_id": "int16", "installments": "int16",
    "merchant_category_id": "int16", "month_lag": "int8",
    "purchase_amount": "float64",  # 金额还原需 float64 精度,还原后再降 float32
    "category_2": "float32", "state_id": "int8", "subsector_id": "int8",
    "authorized_flag": "object", "category_1": "object", "category_3": "object",
    "merchant_id": "object",
}


def load_transactions(name: str, sample_cards=None) -> pd.DataFrame:
    """读取交易表(hist / new),DEBUG 模式下仅保留抽样卡的交易。"""
    path = os.path.join(CONFIG["DATA_DIR"], name)
    df = pd.read_csv(path, dtype=TRANS_DTYPES)
    if sample_cards is not None:
        df = df[df["card_id"].isin(sample_cards)].reset_index(drop=True)
    log(f"  {name}: {df.shape}")
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值分层填充 + 异常值过滤 + 金额脱敏还原(不粗暴删行)。

    业务化处理逻辑(逐字段):
    - authorized_flag / category_1:Y/N -> 1/0(数值化便于聚合出"授权占比");
    - installments:-1 与 999 为哨兵异常(EDA 证实 -1 与 category_3 缺失完全同批),
      转 NaN 交由树模型原生处理,不删行(保留 count 信息);
    - category_3:分期类型 A/B/C 序数编码 0/1/2,缺失填 -1 单独成档;
    - category_2:区域编码 1-5,缺失填 0 作为"未知区域"新档;
    - merchant_id:缺失填哨兵值,保证 nunique 统计口径一致;
    - purchase_amount:按 raddar 发现的线性脱敏逆变换还原真实金额
      real = round(x / 0.00150265118 + 497.06, 2),长尾恢复业务含义;
      再按 99.9% 分位 winsorize 截断大额异常消费(hist 最大值高达 40 亿,属异常单);
    - purchase_date:解析失败/超出赛题时间窗(2017-01 ~ REF_DATE)的无效交易剔除。
    """
    df["authorized_flag"] = (df["authorized_flag"] == "Y").astype(np.int8)
    df["category_1"] = (df["category_1"] == "Y").astype(np.int8)
    df["category_3"] = df["category_3"].map({"A": 0, "B": 1, "C": 2}).fillna(-1).astype(np.int8)
    df["installments"] = df["installments"].replace({-1: np.nan, 999: np.nan}).astype(np.float32)
    df["category_2"] = df["category_2"].fillna(0).astype(np.int8)
    df["merchant_id"] = df["merchant_id"].fillna("M_ID_nan")

    # 金额还原(float64 计算保精度)后 winsorize,再降 float32
    amt = df["purchase_amount"].to_numpy(dtype=np.float64)
    amt = np.round(amt / 0.00150265118 + 497.06, 2)
    upper = np.quantile(amt, 0.999)
    df["purchase_amount"] = np.clip(amt, None, upper).astype(np.float32)

    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    valid = df["purchase_date"].notna() & (df["purchase_date"] >= "2017-01-01") & (df["purchase_date"] < CONFIG["REF_DATE"])
    if (~valid).any():
        df = df[valid].reset_index(drop=True)

    df["card_id"] = df["card_id"].astype("category")  # 加速后续 groupby
    return df


def prep_merchants() -> pd.DataFrame:
    """商户表关联编码:去重 + 序数/标签编码 + 中位数填充,仅保留有效静态属性。

    提分逻辑:merchant_category_id / city / state 等 id 交易表中已有,商户表的
    增量信息在销量/客流的滞后统计与 A-E 分层档位;merchant_id 有 63 条重复须去重,
    avg_purchases_lag* 含 inf 须先转 NaN 再填充,否则聚合统计被污染。
    """
    m = pd.read_csv(os.path.join(CONFIG["DATA_DIR"], "merchants.csv"))
    m = m.drop_duplicates("merchant_id", keep="first")
    keep = ["merchant_id", "merchant_group_id", "numerical_1", "numerical_2",
            "avg_sales_lag3", "avg_purchases_lag3", "avg_sales_lag12", "avg_purchases_lag12",
            "active_months_lag12", "category_4",
            "most_recent_sales_range", "most_recent_purchases_range"]
    m = m[keep]
    m["category_4"] = (m["category_4"] == "Y").astype(np.int8)
    rng_map = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}  # 销量档位 A 最高 -> 序数编码
    m["most_recent_sales_range"] = m["most_recent_sales_range"].map(rng_map).astype(np.int8)
    m["most_recent_purchases_range"] = m["most_recent_purchases_range"].map(rng_map).astype(np.int8)
    for c in ["numerical_1", "numerical_2", "avg_sales_lag3", "avg_purchases_lag3",
              "avg_sales_lag12", "avg_purchases_lag12", "active_months_lag12"]:
        m[c] = m[c].replace([np.inf, -np.inf], np.nan)
        m[c] = m[c].fillna(m[c].median()).astype(np.float32)
    m.columns = ["merchant_id"] + ["mer_" + c for c in m.columns[1:]]
    return reduce_mem_usage(m, verbose=False)


# ============================================================================
# 2. 特征工程层(提分核心)
# ============================================================================
REF_TS = pd.Timestamp(CONFIG["REF_DATE"])

# 巴西主要节日(高分 kernel 通用套路):节前 0-100 天内的"备礼消费"与忠诚度相关
HOLIDAYS = {
    "christmas_2017": "2017-12-25", "mothers_day_2017": "2017-05-14",
    "fathers_day_2017": "2017-08-13", "children_day_2017": "2017-10-12",
    "valentine_2017": "2017-06-12", "black_friday_2017": "2017-11-24",
}


def add_time_features(df: pd.DataFrame, with_holidays: bool) -> pd.DataFrame:
    """时序基础衍生:month_diff 近期性 / 小时 / 周末 / 月份 id / 节日距离 / 组合金额。

    提分逻辑:
    - month_diff = 距参考日月数 + month_lag,衡量"交易发生时距该卡观察期的月数",
      是公开高分方案中单特征贡献最大的近期性度量;
    - duration / amount_month_ratio:金额与近期性的乘除组合,刻画"近期高消费"信号;
    - 节日距离:仅保留节前 100 天窗口内的天数差,窗口外置 0。
    """
    df["pt"] = (df["purchase_date"].astype("int64") // 10 ** 9).astype(np.int64)  # epoch 秒
    df["month_diff"] = ((REF_TS - df["purchase_date"]).dt.days // 30).astype(np.int16) + df["month_lag"]
    df["hour"] = df["purchase_date"].dt.hour.astype(np.int8)
    df["dow"] = df["purchase_date"].dt.dayofweek.astype(np.int8)
    df["weekend"] = (df["dow"] >= 5).astype(np.int8)
    df["month_id"] = (df["purchase_date"].dt.year * 12 + df["purchase_date"].dt.month).astype(np.int16)
    # 金额组合特征
    inst = np.maximum(df["installments"].fillna(1).to_numpy(np.float32), 1)
    df["price"] = (df["purchase_amount"] / inst).astype(np.float32)          # 单期实付
    df["duration"] = (df["purchase_amount"] * df["month_diff"]).astype(np.float32)
    df["amount_month_ratio"] = (df["purchase_amount"] / (df["month_diff"] + 1.0)).astype(np.float32)
    # 类别 one-hot(int8 控内存),聚合时求 mean 即各档占比
    for v in range(6):
        df[f"cat2_{v}"] = (df["category_2"] == v).astype(np.int8)
    for v in (-1, 0, 1, 2):
        df[f"cat3_{v}"] = (df["category_3"] == v).astype(np.int8)
    if with_holidays:
        for name, day in HOLIDAYS.items():
            d = (pd.Timestamp(day) - df["purchase_date"]).dt.days
            df[f"hol_{name}"] = np.where((d >= 0) & (d < 100), d, 0).astype(np.int16)
    return df


def aggregate_transactions(df: pd.DataFrame, prefix: str, mer: pd.DataFrame) -> pd.DataFrame:
    """双交易表分层聚合主函数:card_id 粒度全量统计。

    覆盖任务书要求的 min/max/mean/std/sum/count/25%/75% 分位数/极差,并叠加:
    nunique 高频特征(独特商户数/类目数/活跃月份数)、授权占比、类别档位占比、
    节日距离均值、商户静态属性均值。全部走 cython 聚合路径(自定义 lambda 会退化
    为逐组 python 调用,29M 行上不可接受)。
    """
    df = df.merge(mer, on="merchant_id", how="left")  # 商户静态属性下沉到交易粒度

    agg = {
        "purchase_amount": ["mean", "sum", "std", "min", "max", "median"],
        "price": ["mean", "max"],
        "duration": ["mean", "min", "max"],
        "amount_month_ratio": ["mean", "min", "max"],
        "installments": ["mean", "sum", "max", "std"],
        "month_lag": ["mean", "min", "max", "std"],
        "month_diff": ["mean", "min", "max", "std"],
        "authorized_flag": ["mean"],                      # 授权交易占比
        "category_1": ["mean"],
        "weekend": ["mean"], "hour": ["mean", "std"], "dow": ["mean"],
        "merchant_id": ["nunique"],                       # 用户独特商户数
        "merchant_category_id": ["nunique"],
        "state_id": ["nunique"], "city_id": ["nunique"], "subsector_id": ["nunique"],
        "month_id": ["nunique"],                          # 活跃月份数
        "pt": ["min", "max"],
        "mer_numerical_1": ["mean"], "mer_numerical_2": ["mean"],
        "mer_avg_sales_lag3": ["mean"], "mer_avg_purchases_lag3": ["mean"],
        "mer_avg_sales_lag12": ["mean"], "mer_avg_purchases_lag12": ["mean"],
        "mer_active_months_lag12": ["mean"], "mer_category_4": ["mean"],
        "mer_most_recent_sales_range": ["mean"], "mer_most_recent_purchases_range": ["mean"],
        "mer_merchant_group_id": ["nunique"],
    }
    agg.update({c: ["mean"] for c in df.columns if c.startswith(("cat2_", "cat3_", "hol_"))})

    g = df.groupby("card_id", observed=True)
    out = g.agg(agg)
    out.columns = [f"{prefix}_{c}_{s}" for c, s in out.columns]
    out[f"{prefix}_count"] = g.size()

    # 分位数与极差:单独走 cython quantile,再派生 ptp / IQR
    q = g["purchase_amount"].quantile([0.25, 0.75]).unstack()
    out[f"{prefix}_purchase_amount_q25"] = q[0.25]
    out[f"{prefix}_purchase_amount_q75"] = q[0.75]
    out[f"{prefix}_purchase_amount_iqr"] = q[0.75] - q[0.25]
    out[f"{prefix}_purchase_amount_ptp"] = (
        out[f"{prefix}_purchase_amount_max"] - out[f"{prefix}_purchase_amount_min"])

    # 时间跨度与近期性(recency):距参考日越近的活跃卡忠诚度信号越强
    out[f"{prefix}_date_ptp_days"] = (out[f"{prefix}_pt_max"] - out[f"{prefix}_pt_min"]) / 86400.0
    out[f"{prefix}_last_to_ref_days"] = (REF_TS.timestamp() - out[f"{prefix}_pt_max"]) / 86400.0
    # 消费强度:平均每活跃月交易数 / 金额;CLV 组合(count×sum / 近期性)
    out[f"{prefix}_count_per_month"] = out[f"{prefix}_count"] / (out[f"{prefix}_month_id_nunique"] + 1e-4)
    out[f"{prefix}_sum_per_month"] = out[f"{prefix}_purchase_amount_sum"] / (out[f"{prefix}_month_id_nunique"] + 1e-4)
    out[f"{prefix}_clv"] = out[f"{prefix}_count"] * out[f"{prefix}_purchase_amount_sum"] / (out[f"{prefix}_month_diff_mean"] + 1.0)
    return reduce_mem_usage(out.reset_index(), verbose=False)


def sequence_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """时序间隔特征:按卡按时间排序后的相邻两次交易间隔(天)统计。

    提分逻辑:消费节奏(间隔均值/波动/最大断档)是忠诚度的直接行为学信号,
    通用 baseline 完全缺失;diff 为向量化 groupby-diff,29M 行可承受。
    """
    s = df[["card_id", "pt"]].sort_values(["card_id", "pt"])
    gap = s.groupby("card_id", observed=True)["pt"].diff() / 86400.0
    s = s.assign(gap=gap.astype(np.float32))
    out = s.groupby("card_id", observed=True)["gap"].agg(["mean", "std", "max", "median"])
    out.columns = [f"{prefix}_gap_{c}" for c in out.columns]
    return reduce_mem_usage(out.reset_index(), verbose=False)


def monthly_volatility(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """月度消费波动:先聚到 card×月 的月消费额,再对月序列做二阶统计。

    提分逻辑:忠诚用户月消费稳定(std 低、月均高),波动大者流失风险高。
    """
    m = df.groupby(["card_id", "month_id"], observed=True)["purchase_amount"].sum().rename("msum").reset_index()
    out = m.groupby("card_id", observed=True)["msum"].agg(["mean", "std", "max", "min"])
    out.columns = [f"{prefix}_monthsum_{c}" for c in out.columns]
    return reduce_mem_usage(out.reset_index(), verbose=False)


def trend_features(hist: pd.DataFrame) -> pd.DataFrame:
    """近期趋势:最近 3 个月(month_lag>=-2)与全期的量/额比值。

    提分逻辑:忠诚度评分锚定观察期,近端行为权重远大于远端;
    近 3 月占比刻画消费趋势(上升/衰减),对 outlier(流失样本)判别尤其有效。
    """
    g_all = hist.groupby("card_id", observed=True)["purchase_amount"].agg(["sum", "count"])
    rec = hist[hist["month_lag"] >= -2]
    g_rec = rec.groupby("card_id", observed=True)["purchase_amount"].agg(["sum", "count"])
    out = pd.DataFrame(index=g_all.index)
    out["hist_recent3_count"] = g_rec["count"]
    out["hist_recent3_sum"] = g_rec["sum"]
    out["hist_recent3_count_ratio"] = g_rec["count"] / (g_all["count"] + 1e-4)
    out["hist_recent3_sum_ratio"] = g_rec["sum"] / (g_all["sum"] + 1e-4)
    return reduce_mem_usage(out.reset_index(), verbose=False)


def cross_features(hist: pd.DataFrame) -> pd.DataFrame:
    """商户交叉特征:card × merchant_category / city / installments 二阶组合统计。

    提分逻辑:
    - top1 占比 + 熵:消费是否集中于单一类目/城市 —— 集中度即偏好强度;
    - 分期档位占比(0/1/2/3+):card_id-installments 组合的分布画像;
    全部向量化实现(pair 计数 -> 组内归一 -> 组内 max / 熵),避免 apply。
    """
    out = None
    for key, tag in [("merchant_category_id", "mcat"), ("city_id", "city")]:
        pair = hist.groupby(["card_id", key], observed=True).size().rename("cnt").reset_index()
        tot = pair.groupby("card_id", observed=True)["cnt"].transform("sum")
        p = (pair["cnt"] / tot).to_numpy(np.float64)
        pair["share"] = p
        pair["plogp"] = -p * np.log(p + 1e-12)
        g = pair.groupby("card_id", observed=True)
        f = pd.DataFrame({f"hist_{tag}_top1_share": g["share"].max(),
                          f"hist_{tag}_entropy": g["plogp"].sum()})
        out = f if out is None else out.join(f)
    # 分期档位占比:0=一次付清,1、2、3+ 三档
    b = hist[["card_id", "installments"]].dropna()
    b = b.assign(bucket=np.clip(b["installments"].to_numpy(np.float32), 0, 3).astype(np.int8))
    piv = b.groupby(["card_id", "bucket"], observed=True).size().unstack(fill_value=0)
    piv = piv.div(piv.sum(axis=1), axis=0)
    piv.columns = [f"hist_inst_share_{int(c)}" for c in piv.columns]
    out = out.join(piv)
    return reduce_mem_usage(out.reset_index(), verbose=False)


def month_lag_pivot(hist: pd.DataFrame) -> pd.DataFrame:
    """month_lag 透视 + 月度金额斜率(Top1 逐月序列思想的表格化)。

    提分逻辑:忠诚度锚定观察期,近端(lag 0..-6)逐月的量/额分布与其一阶趋势
    (衰减/上升斜率)比全期统计更能刻画行为轨迹;斜率用 cov/var 公式向量化。
    """
    g = hist.groupby(["card_id", "month_lag"], observed=True)["purchase_amount"] \
            .agg(["count", "sum"]).reset_index()
    g["ml"] = g["month_lag"].clip(lower=-6)  # -6 及更早归并一档
    piv = g.groupby(["card_id", "ml"], observed=True)[["count", "sum"]].sum().unstack(fill_value=0)
    piv.columns = [f"hist_lag{int(l)}_{s}" for s, l in piv.columns]
    grp = g.groupby("card_id", observed=True)
    xm = grp["month_lag"].transform("mean")
    ym = grp["sum"].transform("mean")
    tmp = pd.DataFrame({"card_id": g["card_id"],
                        "num": (g["month_lag"] - xm) * (g["sum"] - ym),
                        "den": (g["month_lag"] - xm) ** 2})
    s = tmp.groupby("card_id", observed=True).sum()
    piv["hist_monthsum_slope"] = s["num"] / (s["den"] + 1e-9)
    return reduce_mem_usage(piv.reset_index(), verbose=False)


def mode_features(hist: pd.DataFrame) -> pd.DataFrame:
    """众数聚合(63rd 方案):最常消费的城市/类目/州,pair 计数 + idxmax 向量化
    (原仓库 groupby.apply(mode) 在 29M 行上不可用)。"""
    out = None
    for col in ["city_id", "merchant_category_id", "state_id"]:
        pair = hist.groupby(["card_id", col], observed=True).size().rename("n").reset_index()
        idx = pair.groupby("card_id", observed=True)["n"].idxmax()
        f = pair.loc[idx, ["card_id", col]].set_index("card_id")
        f.columns = [f"hist_mode_{col}"]
        out = f if out is None else out.join(f)
    return reduce_mem_usage(out.reset_index(), verbose=False)


def seq_embedding(hist: pd.DataFrame, n_dim: int = 8) -> pd.DataFrame:
    """行为序列嵌入(21st 方案 CountVector/word2vec 家族的零依赖实现):
    每卡的类目/商户序列视作文本 -> TFIDF -> TruncatedSVD 低维稠密向量,
    捕捉聚合统计丢失的"逛店组合"信息。"""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    out = None
    for col, tag, vocab in [("merchant_category_id", "mcat", 400),
                            ("merchant_id", "mid", 20000)]:
        seq = hist.groupby("card_id", observed=True)[col].agg(
            lambda x: " ".join(map(str, x.astype(str))))
        tf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=vocab)
        emb = TruncatedSVD(n_components=n_dim, random_state=2019).fit_transform(
            tf.fit_transform(seq.values))
        f = pd.DataFrame(emb, index=seq.index,
                         columns=[f"hist_{tag}_svd{i}" for i in range(n_dim)])
        out = f if out is None else out.join(f)
    return reduce_mem_usage(out.reset_index(), verbose=False)



def card_side_features(df: pd.DataFrame) -> pd.DataFrame:
    """train/test 卡片侧特征:开卡时长与匿名属性组合。"""
    df = df.copy()
    fam = pd.to_datetime(df["first_active_month"], format="%Y-%m")
    fam = fam.fillna(fam.mode()[0])  # test 有 1 条缺失,众数填充
    df["fam_ts"] = (fam.astype("int64") // 10 ** 9).astype(np.int64)
    df["fam_month_id"] = (fam.dt.year * 12 + fam.dt.month).astype(np.int16)
    df["elapsed_days"] = ((REF_TS - fam).dt.days).astype(np.int32)  # 开卡至今
    df["feature_sum"] = (df["feature_1"] + df["feature_2"] + df["feature_3"]).astype(np.int8)
    for c in ("feature_1", "feature_2", "feature_3"):
        df[f"days_x_{c}"] = (df["elapsed_days"] * df[c]).astype(np.int32)
    return df.drop(columns=["first_active_month"])


def ratio_features(feat: pd.DataFrame) -> pd.DataFrame:
    """hist ↔ new 跨表比值/差值 + 开卡衔接特征(在合表后计算)。

    提分逻辑:new 表是"观察期后在新商户的消费",new/hist 的量、额、商户数比值
    直接度量"忠诚度的增量行为",是与 target 定义最贴近的一组特征。
    """
    eps = 1e-4
    feat["new_hist_count_ratio"] = feat["new_count"] / (feat["hist_count"] + eps)
    feat["new_hist_sum_ratio"] = feat["new_purchase_amount_sum"] / (feat["hist_purchase_amount_sum"] + eps)
    feat["new_hist_mean_diff"] = feat["new_purchase_amount_mean"] - feat["hist_purchase_amount_mean"]
    feat["new_hist_merchant_ratio"] = feat["new_merchant_id_nunique"] / (feat["hist_merchant_id_nunique"] + eps)
    # 开卡 -> 首笔/末笔交易的衔接时长(天)
    feat["hist_first_buy_days"] = (feat["hist_pt_min"] - feat["fam_ts"]) / 86400.0
    feat["new_first_buy_days"] = (feat["new_pt_min"] - feat["fam_ts"]) / 86400.0
    # hist 末笔 -> new 首笔的空窗
    feat["hist_to_new_gap_days"] = (feat["new_pt_min"] - feat["hist_pt_max"]) / 86400.0
    # 观察月锚定派生:开卡到观察月的月数 / 观察月距数据末端的月数
    feat["ref_minus_fam_months"] = feat["hist_ref_month_id"] - feat["fam_month_id"]
    feat["ref_to_end_months"] = (2018 * 12 + 5) - feat["hist_ref_month_id"]
    # new×hist 系统性算术交叉(21st 方案:先生成 -、/ 再靠重要性筛选淘汰)
    for c in ["purchase_amount_max", "purchase_amount_std", "month_diff_mean",
              "installments_mean", "duration_mean", "count_per_month"]:
        feat[f"x_{c}_diff"] = feat[f"new_{c}"] - feat[f"hist_{c}"]
        feat[f"x_{c}_ratio"] = feat[f"new_{c}"] / (feat[f"hist_{c}"] + 1e-4)
    return feat


def build_features() -> pd.DataFrame:
    """特征构建总入口:读取 -> 清洗 -> 三路聚合 -> 合表 -> 缓存 parquet。"""
    log("读取 train/test ...")
    train = pd.read_csv(os.path.join(CONFIG["DATA_DIR"], "train.csv"))
    test = pd.read_csv(os.path.join(CONFIG["DATA_DIR"], "test.csv"))

    sample_cards = None
    if CONFIG["DEBUG"]:
        rng = np.random.RandomState(CONFIG["SEED"])
        train = train.sample(CONFIG["DEBUG_CARDS"], random_state=rng).reset_index(drop=True)
        test = test.sample(CONFIG["DEBUG_CARDS"] // 2, random_state=rng).reset_index(drop=True)
        sample_cards = set(train["card_id"]) | set(test["card_id"])
        log(f"DEBUG 模式:抽样 {len(sample_cards)} 张卡")

    mer = prep_merchants()
    log(f"merchants 处理完成: {mer.shape}")

    # ---- historical(含 declined 子集)----
    hist = load_transactions("historical_transactions.csv", sample_cards)
    hist = clean_transactions(hist)
    hist = add_time_features(hist, with_holidays=True)
    log("hist 聚合 ...")
    feats = [aggregate_transactions(hist, "hist", mer)]
    feats.append(sequence_features(hist, "hist"))
    feats.append(monthly_volatility(hist, "hist"))
    feats.append(trend_features(hist))
    feats.append(cross_features(hist))
    feats.append(month_lag_pivot(hist))
    feats.append(mode_features(hist))
    feats.append(seq_embedding(hist))
    # 观察月锚定(1st place 核心思路):month_id - month_lag 即各卡参考月,理论上恒定,
    # target 分布与观察月强相关;max 兼容极少数跨月脏数据
    hist["ref_month_id"] = (hist["month_id"] - hist["month_lag"]).astype(np.int16)
    r = hist.groupby("card_id", observed=True)["ref_month_id"].agg(["max"])
    r.columns = ["hist_ref_month_id"]
    feats.append(reduce_mem_usage(r.reset_index(), verbose=False))
    # 被拒交易单独小规模聚合:风险/摩擦信号
    dec = hist[hist["authorized_flag"] == 0]
    d = dec.groupby("card_id", observed=True).agg(
        {"purchase_amount": ["mean", "sum"], "month_lag": ["mean"], "merchant_id": ["nunique"]})
    d.columns = [f"dec_{c}_{s}" for c, s in d.columns]
    d["dec_count"] = dec.groupby("card_id", observed=True).size()
    feats.append(reduce_mem_usage(d.reset_index(), verbose=False))
    del hist, dec, d
    gc.collect()

    # ---- new merchant ----
    new = load_transactions("new_merchant_transactions.csv", sample_cards)
    new = clean_transactions(new)
    new = add_time_features(new, with_holidays=False)
    log("new 聚合 ...")
    feats.append(aggregate_transactions(new, "new", mer))
    feats.append(monthly_volatility(new, "new"))
    # new 表按 month_lag=1/2 拆分:观察期后第 1/2 个月行为分别刻画(Top 方案通用)
    for lag in (1, 2):
        sub = new[new["month_lag"] == lag]
        a = sub.groupby("card_id", observed=True).agg(
            {"purchase_amount": ["count", "sum", "mean"], "merchant_id": ["nunique"]})
        a.columns = [f"newlag{lag}_{c}_{s}" for c, s in a.columns]
        feats.append(reduce_mem_usage(a.reset_index(), verbose=False))
    del new, mer
    gc.collect()

    # ---- 合表 ----
    log("合并特征 ...")
    base = pd.concat([train.assign(is_train=1), test.assign(is_train=0, target=np.nan)],
                     ignore_index=True)
    base = card_side_features(base)
    for f in feats:
        f["card_id"] = f["card_id"].astype(str)
        base = base.merge(f, on="card_id", how="left")
    base = ratio_features(base)
    base = reduce_mem_usage(base)

    os.makedirs(CONFIG["PROC_DIR"], exist_ok=True)
    cache = os.path.join(CONFIG["PROC_DIR"],
                         "features_debug.parquet" if CONFIG["DEBUG"] else "features.parquet")
    base.to_parquet(cache)
    log(f"特征缓存至 {cache}: {base.shape}")
    return base


# ============================================================================
# 3. 特征筛选层
# ============================================================================
def select_features(X: pd.DataFrame, y: pd.Series, feat_cols: list) -> tuple:
    """方差过滤 + LGB gain 重要性筛选,返回(保留列, 重要性表)。

    提分逻辑:聚合特征里的近常数/零贡献列会在高维稀疏下加剧过拟合;
    先去零方差,再用 3 折快速 LGB 的平均 gain 排序取 TOP_K,兼顾降维与稳健。
    本仓实测:251 列 → 方差过滤 251(未删)→ gain>0 取 250 列(TOP_K=300 未触顶)。
    """
    keep = [c for c in feat_cols if X[c].nunique(dropna=False) > 1]  # 方差过滤
    log(f"方差过滤: {len(feat_cols)} -> {len(keep)}")

    params = dict(objective="regression", metric="rmse", learning_rate=0.05,
                  num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  min_data_in_leaf=100, verbosity=-1, num_threads=CONFIG["N_THREADS"],
                  seed=CONFIG["SEED"])
    imp = np.zeros(len(keep))
    skf = StratifiedKFold(3, shuffle=True, random_state=CONFIG["SEED"])
    strat = (y < -30).astype(int)
    for tr, va in skf.split(X[keep], strat):
        ds_tr = lgb.Dataset(X.iloc[tr][keep], y.iloc[tr])
        ds_va = lgb.Dataset(X.iloc[va][keep], y.iloc[va])
        m = lgb.train(params, ds_tr, 2000, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        imp += m.feature_importance("gain")
    imp_df = pd.DataFrame({"feature": keep, "gain": imp / 3}).sort_values("gain", ascending=False)
    selected = imp_df[imp_df["gain"] > 0].head(CONFIG["TOP_K"])["feature"].tolist()
    log(f"重要性筛选: {len(keep)} -> {len(selected)} (TOP_K={CONFIG['TOP_K']})")
    return selected, imp_df


# ============================================================================
# 4. 验证与训练层
# ============================================================================
def make_folds(y: pd.Series):
    """分层 10 折:target 连续无法直接分层,按 outlier(<-30)二值分层,
    保证各折 outlier 比例一致 —— 本赛 CV/LB 相关性的关键。折划分全模型共享。"""
    skf = StratifiedKFold(CONFIG["N_FOLDS"], shuffle=True, random_state=CONFIG["SEED"])
    return list(skf.split(np.zeros(len(y)), (y < -30).astype(int)))


# 预调优超参(Optuna TPE 搜索 + Top 方案经验收敛值;TUNE=True 可复搜)
LGB_PARAMS = dict(
    objective="regression", metric="rmse", boosting="gbdt",
    learning_rate=0.01, num_leaves=63, max_depth=8, min_data_in_leaf=150,
    feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1,
    lambda_l1=1.0, lambda_l2=10.0,  # 加大正则抑制高维过拟合
    verbosity=-1, num_threads=CONFIG["N_THREADS"], seed=CONFIG["SEED"])

XGB_PARAMS = dict(
    n_estimators=10000, learning_rate=0.01, max_depth=7, min_child_weight=60,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=10.0,
    tree_method="hist", eval_metric="rmse", early_stopping_rounds=200,
    n_jobs=CONFIG["N_THREADS"], random_state=CONFIG["SEED"], verbosity=0)

CAT_PARAMS = dict(
    iterations=12000, learning_rate=0.02, depth=8, l2_leaf_reg=12.0,
    bootstrap_type="Bernoulli", subsample=0.8, loss_function="RMSE",
    random_seed=CONFIG["SEED"], thread_count=CONFIG["N_THREADS"],
    allow_writing_files=False, verbose=0)

CLF_PARAMS = dict(  # outlier 二分类:概率作为 Stacking 元特征
    objective="binary", metric="auc", learning_rate=0.02, num_leaves=31,
    max_depth=7, min_data_in_leaf=100, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l1=1.0, lambda_l2=10.0,
    is_unbalance=True, verbosity=-1, num_threads=CONFIG["N_THREADS"], seed=CONFIG["SEED"])

# huber 目标的 LGB:对 -33.22 哨兵 outlier 的梯度饱和,残差与 L2 系模型互补
HUB_PARAMS = {**LGB_PARAMS, "objective": "huber", "alpha": 1.35}


def _rounds(default: int) -> int:
    return 500 if CONFIG["DEBUG"] else default


def cv_lightgbm(X, y, X_test, folds, params, label="lgb"):
    """LightGBM 十折 CV:早停 + 固定种子,输出 OOF / test 均值 / 逐折 RMSE / gain。"""
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    gain = np.zeros(X.shape[1])
    scores = []
    for k, (tr, va) in enumerate(folds):
        m = lgb.train(params, lgb.Dataset(X.iloc[tr], y.iloc[tr]),
                      _rounds(10000), valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
        gain += m.feature_importance("gain") / len(folds)
        scores.append(float(np.sqrt(mean_squared_error(y.iloc[va], oof[va]))))
        log(f"  [{label}] fold{k + 1}: RMSE={scores[-1]:.5f} (iter={m.best_iteration})")
    return oof, pred, scores, gain


def cv_xgboost(X, y, X_test, folds):
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    scores = []
    params = {**XGB_PARAMS, "n_estimators": _rounds(10000)}
    for k, (tr, va) in enumerate(folds):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        oof[va] = m.predict(X.iloc[va])
        pred += m.predict(X_test) / len(folds)
        scores.append(float(np.sqrt(mean_squared_error(y.iloc[va], oof[va]))))
        log(f"  [xgb] fold{k + 1}: RMSE={scores[-1]:.5f} (iter={m.best_iteration})")
    return oof, pred, scores


def cv_catboost(X, y, X_test, folds):
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    scores = []
    params = {**CAT_PARAMS, "iterations": _rounds(12000)}
    for k, (tr, va) in enumerate(folds):
        m = CatBoostRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]),
              early_stopping_rounds=200, use_best_model=True)
        oof[va] = m.predict(X.iloc[va])
        pred += m.predict(X_test) / len(folds)
        scores.append(float(np.sqrt(mean_squared_error(y.iloc[va], oof[va]))))
        log(f"  [cat] fold{k + 1}: RMSE={scores[-1]:.5f} (iter={m.get_best_iteration()})")
    return oof, pred, scores


def cv_outlier_clf(X, y, X_test, folds):
    """outlier 二分类(P(target=-33.22)):概率进入二层 Stacking。

    提分逻辑:1.09% 的 -33.22 哨兵样本贡献了 RMSE 的主要部分,回归模型难以
    单独刻画;显式建模 outlier 概率让元模型学会"高危卡往低调"的修正。
    """
    ybin = pd.Series((y < -30).astype(int), index=y.index)
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds):
        m = lgb.train(CLF_PARAMS, lgb.Dataset(X.iloc[tr], ybin.iloc[tr]),
                      _rounds(10000), valid_sets=[lgb.Dataset(X.iloc[va], ybin.iloc[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
    auc = roc_auc_score(ybin, oof)
    log(f"  [clf] outlier AUC={auc:.5f}")
    return oof, pred, auc


# ---------------------------------------------------------------------------
# Optuna 贝叶斯(TPE)超参搜索:TUNE=True 时对三类模型分别复搜
# ---------------------------------------------------------------------------
def tune_model(model_type: str, X, y, n_trials: int) -> dict:
    """3 折子代理快速搜索(全折搜索代价过高),搜索空间即推荐调参范围。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds3 = list(StratifiedKFold(3, shuffle=True, random_state=CONFIG["SEED"])
                  .split(X, (y < -30).astype(int)))

    def objective(trial):
        if model_type == "lgb":
            p = {**LGB_PARAMS,
                 "learning_rate": 0.02,
                 "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                 "max_depth": trial.suggest_int("max_depth", 6, 11),
                 "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
                 "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
                 "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                 "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
                 "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 30, log=True)}
        rmses = []
        for tr, va in folds3:
            if model_type == "lgb":
                m = lgb.train(p, lgb.Dataset(X.iloc[tr], y.iloc[tr]), 3000,
                              valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                pv = m.predict(X.iloc[va], num_iteration=m.best_iteration)
            elif model_type == "xgb":
                m = xgb.XGBRegressor(**{**XGB_PARAMS, "n_estimators": 3000, "learning_rate": 0.02,
                    "max_depth": trial.suggest_int("max_depth", 5, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 10, 150),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-1, 30, log=True)})
                m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
                pv = m.predict(X.iloc[va])
            else:  # cat
                m = CatBoostRegressor(**{**CAT_PARAMS, "iterations": 3000, "learning_rate": 0.03,
                    "depth": trial.suggest_int("depth", 5, 10),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0)})
                m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]),
                      early_stopping_rounds=100, use_best_model=True)
                pv = m.predict(X.iloc[va])
            rmses.append(np.sqrt(mean_squared_error(y.iloc[va], pv)))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=CONFIG["SEED"]))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log(f"  [tune:{model_type}] best RMSE={study.best_value:.5f} params={study.best_params}")
    return study.best_params


# ============================================================================
# 5. 融合层
# ============================================================================
def stack_ridge(oofs: dict, preds: dict, y: pd.Series, folds):
    """二层 Stacking:元特征 =(lgb, xgb, cat OOF)+ outlier 概率,元模型 Ridge。

    提分逻辑:线性元模型在强相关基学习器上最稳(不易过拟合 OOF 噪声);
    outlier 概率列让二层学到"高危卡额外下调"的线性修正。折划分与一层一致,
    对训练侧同样产出 OOF,保证汇报的 stacking RMSE 无泄漏。
    """
    meta_X = np.column_stack([oofs[k] for k in sorted(oofs)])
    meta_T = np.column_stack([preds[k] for k in sorted(preds)])
    oof_st = np.zeros(len(y))
    pred_st = np.zeros(meta_T.shape[0])
    coefs = []
    for tr, va in folds:
        r = Ridge(alpha=1.0, random_state=CONFIG["SEED"])
        r.fit(meta_X[tr], y.iloc[tr])
        oof_st[va] = r.predict(meta_X[va])
        pred_st += r.predict(meta_T) / len(folds)
        coefs.append(r.coef_)
    rmse = float(np.sqrt(mean_squared_error(y, oof_st)))
    log(f"Stacking(Ridge) OOF RMSE={rmse:.5f}  系数均值={np.mean(coefs, 0).round(3)}")
    return oof_st, pred_st, rmse


def blend_weighted(oofs: dict, preds: dict, y: pd.Series):
    """备选方案:OOF 上数值优化的非负加权融合(权重和为 1)。"""
    keys = sorted(k for k in oofs if k != "clf")
    P = np.column_stack([oofs[k] for k in keys])
    T = np.column_stack([preds[k] for k in keys])

    def loss(w):
        return np.sqrt(mean_squared_error(y, P @ w))

    w0 = np.ones(len(keys)) / len(keys)
    res = minimize(loss, w0, method="SLSQP", bounds=[(0, 1)] * len(keys),
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w = res.x
    rmse = float(loss(w))
    log(f"加权融合 OOF RMSE={rmse:.5f}  权重={dict(zip(keys, w.round(3)))}")
    return P @ w, T @ w, rmse, dict(zip(keys, w.round(4).tolist()))


def postprocess_outlier(pred_test, clf_prob_test, pred_full_test, top_n=20000):
    """可选后处理(消融用):对 outlier 概率最高的 top_n 张测试卡,
    用受 outlier 拉动更充分的全量模型预测替换融合预测。
    注意:该操作对 Public LB 常有小幅收益,但对 Private 有风险,默认不启用。"""
    idx = np.argsort(-clf_prob_test)[:top_n]
    out = pred_test.copy()
    out[idx] = pred_full_test[idx]
    return out


# ============================================================================
# 6. 产物保存层
# ============================================================================
def save_submission(card_ids, pred, name):
    path = os.path.join(CONFIG["OUT_DIR"], f"submission_{name}.csv")
    pd.DataFrame({"card_id": card_ids, "target": pred}).to_csv(path, index=False)
    log(f"保存 {path}")


def save_base_member(name, oof, pred):
    """基础世代成员统一落盘为 outputs/base/{name}.npz。

    后续的 hetero / target_encoding / fusion / fuse_final 都依赖这套协议:
    每个成员至少提供 `oof` 与 `pred(test)` 两个数组。
    """
    base_dir = os.path.join(CONFIG["OUT_DIR"], "base")
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{name}.npz")
    np.savez(path, oof=oof, pred=pred)
    log(f"保存 {path}")


def plot_importance(imp_df: pd.DataFrame, selected: list):
    """特征重要性 Top-40 水平条形图(单序列/浅色表面/细条/克制网格)。"""
    top = imp_df[imp_df["feature"].isin(selected)].head(40).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 12), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh(top["feature"], top["gain"], color="#2a78d6", height=0.62)
    ax.set_title("LightGBM Feature Importance (gain, Top 40)", loc="left",
                 fontsize=12, color="#0b0b0b", pad=12)
    ax.tick_params(colors="#898781", labelsize=7.5)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    fig.tight_layout()
    path = os.path.join(CONFIG["OUT_DIR"], "feature_importance.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    log(f"保存 {path}")


# ============================================================================
# 7. 主流程
# ============================================================================
def main():
    np.random.seed(CONFIG["SEED"])
    os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)

    # ---- 特征(带缓存:生成一次,反复实验;DEBUG 缓存独立命名互不污染)----
    cache = os.path.join(CONFIG["PROC_DIR"],
                         "features_debug.parquet" if CONFIG["DEBUG"] else "features.parquet")
    if os.path.exists(cache):
        base = pd.read_parquet(cache)
        log(f"读取特征缓存 {cache}: {base.shape}")
    else:
        base = build_features()

    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    drop_cols = {"card_id", "target", "is_train"}
    feat_cols = [c for c in train.columns if c not in drop_cols]
    log(f"train={train.shape} test={test.shape} 特征数={len(feat_cols)}")

    # ---- 特征筛选 ----
    selected, imp_df = select_features(train, y, feat_cols)
    imp_df.to_csv(os.path.join(CONFIG["OUT_DIR"], "feature_importance.csv"), index=False)
    X, X_test = train[selected], test[selected]

    # ---- 可选:Optuna 贝叶斯调参 ----
    if CONFIG["TUNE"]:
        log("Optuna TPE 调参(3 折代理)...")
        LGB_PARAMS.update(tune_model("lgb", X, y, CONFIG["TUNE_TRIALS"]))
        XGB_PARAMS.update(tune_model("xgb", X, y, CONFIG["TUNE_TRIALS"]))
        CAT_PARAMS.update(tune_model("cat", X, y, CONFIG["TUNE_TRIALS"]))

    # ---- 分层 10 折训练:三回归 + outlier 二分类 ----
    folds = make_folds(y)
    log("训练 LightGBM ...")
    oof_lgb, pred_lgb, sc_lgb, gain = cv_lightgbm(X, y, X_test, folds, LGB_PARAMS)
    log("训练 XGBoost ...")
    oof_xgb, pred_xgb, sc_xgb = cv_xgboost(X, y, X_test, folds)
    log("训练 CatBoost ...")
    oof_cat, pred_cat, sc_cat = cv_catboost(X, y, X_test, folds)
    log("训练 huber-LGB ...")
    oof_hub, pred_hub, sc_hub, _ = cv_lightgbm(X, y, X_test, folds, HUB_PARAMS, "hub")
    log("训练 outlier 分类器 ...")
    oof_clf, pred_clf, auc = cv_outlier_clf(X, y, X_test, folds)

    rmse_of = lambda o: float(np.sqrt(mean_squared_error(y, o)))
    summary = {"lgb": {"folds": sc_lgb, "oof": rmse_of(oof_lgb)},
               "xgb": {"folds": sc_xgb, "oof": rmse_of(oof_xgb)},
               "cat": {"folds": sc_cat, "oof": rmse_of(oof_cat)},
               "hub": {"folds": sc_hub, "oof": rmse_of(oof_hub)},
               "clf_auc": auc}
    for k in ("lgb", "xgb", "cat", "hub"):
        log(f"{k.upper()} OOF RMSE = {summary[k]['oof']:.5f}")

    # ---- 融合:Ridge Stacking(主)+ 加权融合(备选)----
    oofs = {"lgb": oof_lgb, "xgb": oof_xgb, "cat": oof_cat, "hub": oof_hub, "clf": oof_clf}
    preds = {"lgb": pred_lgb, "xgb": pred_xgb, "cat": pred_cat, "hub": pred_hub, "clf": pred_clf}
    oof_st, pred_st, rmse_st = stack_ridge(oofs, preds, y, folds)
    oof_bl, pred_bl, rmse_bl, weights = blend_weighted(oofs, preds, y)
    summary["stack_oof"] = rmse_st
    summary["blend_oof"] = rmse_bl
    summary["blend_weights"] = weights

    # ---- 可选:无 outlier 干净模型 + top-N 后处理(消融)----
    if CONFIG["TRAIN_CLEAN_MODEL"]:
        mask = y > -30
        Xc, yc = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
        folds_c = list(StratifiedKFold(CONFIG["N_FOLDS"], shuffle=True,
                       random_state=CONFIG["SEED"]).split(Xc, np.zeros(len(yc), int)))
        _, pred_clean, sc_clean, _ = cv_lightgbm(Xc, yc, X_test, folds_c, LGB_PARAMS, "clean")
        summary["clean_folds"] = sc_clean
        pred_pp = postprocess_outlier(pred_clean, pred_clf, pred_st)
        save_submission(test["card_id"], pred_pp, "clean_postprocess")

    # ---- 产物保存 ----
    oof_out = pd.DataFrame({"card_id": train["card_id"], "target": y,
                            "oof_lgb": oof_lgb, "oof_xgb": oof_xgb, "oof_cat": oof_cat,
                            "oof_hub": oof_hub, "oof_clf_prob": oof_clf,
                            "oof_stack": oof_st, "oof_blend": oof_bl})
    oof_out.to_csv(os.path.join(CONFIG["OUT_DIR"], "oof_predictions.csv"), index=False)
    # outlier 分类器测试概率单独落盘,供后处理实验复用
    pd.DataFrame({"card_id": test["card_id"], "clf_prob": pred_clf}).to_csv(
        os.path.join(CONFIG["OUT_DIR"], "test_clf_prob.csv"), index=False)
    for name, oof, pred in [("lgb", oof_lgb, pred_lgb), ("xgb", oof_xgb, pred_xgb),
                            ("cat", oof_cat, pred_cat), ("hub", oof_hub, pred_hub),
                            ("clf", oof_clf, pred_clf)]:
        save_base_member(name, oof, pred)
    for name, p in [("lgb", pred_lgb), ("xgb", pred_xgb), ("cat", pred_cat),
                    ("stack", pred_st), ("blend", pred_bl)]:
        save_submission(test["card_id"], p, name)
    plot_importance(imp_df, selected)
    with open(os.path.join(CONFIG["OUT_DIR"], "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"完成。Stacking OOF RMSE = {rmse_st:.5f}(建议提交 submission_stack.csv)")


if __name__ == "__main__":
    main()
