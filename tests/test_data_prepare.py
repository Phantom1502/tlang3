"""
tests/test_data_prepare.py

Test cho app/data_prepare/generator.py (ZoneGenerator) va
app/data_prepare/dataset_builder.py (DatasetBuilder, augment_shift).
Style giong tests/testlang.py cua project (plain assert, khong dung pytest).

Trong tam: T1-T3 la regression test cho 2 bug da fix (seed bi reset moi
lan goi build_pretrain_rows(), va leaf_recipe bi gan nham full_text) --
neu ai vo tinh regress lai 2 bug nay, test se FAIL ngay.

Chay: python -m tests.test_data_prepare
"""
from app.config.schema import AppConfig, BaseConfig, DataGenV2Config, ModelsConfig, WindowConfig
from app.candle import Candle
from app.data_prepare.dataset_builder import DatasetBuilder, augment_shift
from app.data_prepare.generator import LEAF_RECIPES, ZoneGenerator
from app.lang.parser import Parser

results = []


def check(name: str, condition: bool) -> None:
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


# =====================================================================
# Config test-only, giong tests/testlang.py -- input_candles nho de test
# nhanh, khong lien quan configs/base.yaml that.
# =====================================================================
N_CANDLES = 5

base_cfg = BaseConfig(
    bin_min=0,
    bin_max=1023,
    n_bins=1024,
    zone_width_min_bins=5,
    zone_width_max_bins=20,
    digit_pad=4,
    rr_min=1,
    rr_max=9,
    action_types=("BUY", "SELL", "HOLD"),
    trend_values=("UP", "DOWN", "RANGE"),
)
window_cfg = WindowConfig(input_candles=N_CANDLES, outcome_horizon=N_CANDLES, window_size=2 * N_CANDLES)
models_cfg = ModelsConfig(tokenizer_repo="test-tokenizer", max_position_embeddings=512, presets={})
datagen_cfg = DataGenV2Config(stride=1, n_augments_per_window=0)

cfg = AppConfig(
    base=base_cfg,
    window=window_cfg,
    scales=[],
    models=models_cfg,
    training_defaults={},
    datagen_v2=datagen_cfg,
    rounds={},
)


def make_candles(closes):
    """closes: list gia Close, do dai >= N_CANDLES. Margin +-5 quanh moi
    Close giu cho zone/candle khong cham bien [bin_min, bin_max]."""
    return [Candle(open=c - 2, high=c + 5, low=c - 5, close=c) for c in closes]


BASE_CLOSES = [500, 505, 503, 507, 512]


# =====================================================================
# T1 [REGRESSION - BUG 1]: goi build_pretrain_rows() NHIEU LAN tren CUNG
# 1 DatasetBuilder instance voi CUNG 1 chart -- mo phong dung cach
# ZonePretrainOneFileGen.__iter__ dung 1 builder xuyen suot nhieu window.
#
# Neu bug "ZoneGenerator bi tao moi + seed reset moi lan goi" con ton tai,
# 2 lan goi lien tiep voi CUNG input se ra KET QUA GIONG HET NHAU (RNG luon
# bat dau lai tu dau). Sau khi fix (zone_gen tao 1 lan trong __init__), 2
# lan goi PHAI ra khac nhau (RNG da tien them tu lan goi truoc).
# =====================================================================
builder = DatasetBuilder(cfg, seed=7)
chart = make_candles(BASE_CLOSES)

rows1 = builder.build_pretrain_rows(chart, samples_per_chart=8, n_augments=0)
rows2 = builder.build_pretrain_rows(chart, samples_per_chart=8, n_augments=0)

check("T1a: build_pretrain_rows() sinh duoc sample o ca 2 lan goi", len(rows1) > 0 and len(rows2) > 0)
check(
    "T1b [REGRESSION BUG 1]: 2 lan goi lien tiep tren CUNG 1 builder+chart "
    "phai ra KHAC NHAU (RNG khong bi reset ve seed ban dau)",
    rows1 != rows2,
)

# --- T1c: doi lai, tao 2 builder MOI cung seed=7 -> lan goi DAU TIEN cua
# moi builder phai GIONG NHAU (xac nhan van con deterministic theo seed,
# khong phai random that khong kiem soat duoc) ---
builder_a = DatasetBuilder(cfg, seed=7)
builder_b = DatasetBuilder(cfg, seed=7)
rows_a = builder_a.build_pretrain_rows(chart, samples_per_chart=8, n_augments=0)
rows_b = builder_b.build_pretrain_rows(chart, samples_per_chart=8, n_augments=0)
check("T1c: 2 builder MOI cung seed -> lan goi dau tien giong nhau (van deterministic)", rows_a == rows_b)


# =====================================================================
# T2 [REGRESSION - BUG 2]: leaf_recipe phai la label NGAN "{trend}|{side}",
# KHONG PHAI toan bo prompt+completion.
# =====================================================================
zone_gen = ZoneGenerator(cfg, seed=1)
sample = zone_gen.generate_one(make_candles(BASE_CLOSES))
check("T2a: generate_one sinh duoc 1 sample", sample is not None)
if sample is not None:
    expected_labels = {f"{t}|{s}" for t, s in LEAF_RECIPES}
    check(
        f"T2b [REGRESSION BUG 2]: leaf_recipe ({sample.leaf_recipe!r}) la label ngan hop le, "
        f"KHONG phai full_text",
        sample.leaf_recipe in expected_labels,
    )
    check("T2c: leaf_recipe ngan hon nhieu so voi completion (khong phai full_text)",
          len(sample.leaf_recipe) < len(sample.completion))


# =====================================================================
# T3: bat bien huong zone -- support: lower_bin <= current_price;
# resistance: upper_bin >= current_price. Test truc tiep _pick_zone qua
# nhieu lan random de khong phu thuoc may-rui cua generate_one.
# =====================================================================
zg = ZoneGenerator(cfg, seed=99)
current_price = 500
n_checked_support = 0
n_checked_resistance = 0
for _ in range(300):
    z = zg._pick_zone("support", current_price)
    if z is not None:
        n_checked_support += 1
        if z.lower_bin > current_price:
            check(f"T3a: zone_support.lower_bin ({z.lower_bin}) <= current_price ({current_price})", False)
            break
else:
    check(f"T3a: {n_checked_support} zone_support sinh ra deu co lower_bin <= current_price", True)

for _ in range(300):
    z = zg._pick_zone("resistance", current_price)
    if z is not None:
        n_checked_resistance += 1
        if z.upper_bin < current_price:
            check(f"T3b: zone_resistance.upper_bin ({z.upper_bin}) >= current_price ({current_price})", False)
            break
else:
    check(f"T3b: {n_checked_resistance} zone_resistance sinh ra deu co upper_bin >= current_price", True)

check("T3c: co it nhat vai zone hop le sinh ra (khong phai toan bo bi None do bien)",
      n_checked_support > 50 and n_checked_resistance > 50)


# =====================================================================
# T4: augment_shift -- case binh thuong tra ve chart da dich, case bao hoa
# bien (candle da chiem het [0, n_bins-1]) tra ve None.
# =====================================================================
import random as _random_module

rng = _random_module.Random(0)
chart_mid = make_candles(BASE_CLOSES)   # nam giua [0,1023], con nhieu cho de dich
shifted = augment_shift(chart_mid, rng, n_bins=cfg.base.n_bins)
check("T4a: augment_shift binh thuong -> tra ve chart moi (khong None)", shifted is not None)
if shifted is not None:
    delta = shifted[0].open - chart_mid[0].open
    check("T4b: augment_shift dich DEU moi truong (delta giong nhau tren ca 4 field/moi nen)",
          all(
              (s.open - c.open) == delta and (s.high - c.high) == delta
              and (s.low - c.low) == delta and (s.close - c.close) == delta
              for s, c in zip(shifted, chart_mid)
          ))

chart_full_range = [Candle(open=0, high=cfg.base.bin_max, low=0, close=cfg.base.bin_max)] * N_CANDLES
shifted_full = augment_shift(chart_full_range, rng, n_bins=cfg.base.n_bins)
check("T4c: augment_shift khi da chiem het bien -> tra ve None", shifted_full is None)


# =====================================================================
# T5: build_pretrain_rows() -- schema dung {"prompt","completion"}, va
# MOI row parse well-formed (double-check end-to-end, khong chi tin generator).
# =====================================================================
builder5 = DatasetBuilder(cfg, seed=123)
rows5 = builder5.build_pretrain_rows(make_candles(BASE_CLOSES), samples_per_chart=6, n_augments=2)
check("T5a: build_pretrain_rows sinh duoc it nhat 1 row", len(rows5) > 0)
check("T5b: moi row dung schema {'prompt','completion'}", all(set(r.keys()) == {"prompt", "completion"} for r in rows5))

fail_count = 0
for r in rows5:
    full_text = r["prompt"] + " " + r["completion"]
    pr = Parser.from_text(cfg, full_text).parse()
    if not pr.is_well_formed():
        fail_count += 1
check(f"T5c: {len(rows5) - fail_count}/{len(rows5)} row well-formed (moi row PHAI well-formed)", fail_count == 0)


print("=" * 70)
total = len(results)
passed = sum(1 for _, ok in results if ok)
print(f"TONG: {total} test, {passed} PASS, {total - passed} FAIL")
if passed != total:
    raise SystemExit(1)