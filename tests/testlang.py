"""
tests/test_lang_task1.py

Test cho app/lang (task1: chart_block + think_block, KHONG co action).
Style giong tests/test1.py / tests/test3.py cua project (plain assert,
khong dung pytest).

Gom 3 nhom:
  1. Hanh vi dung mong doi (T1-T8)  -> phai PASS, neu FAIL la regression that.
  2. Gap/leftover DA BIET (T9-T12)  -> PASS = xac nhan van con ton tai o
     trang thai hien tai. Khi ban fix/dop dep, cac test nay se FAIL --
     dung nhu y dinh, no nhac ban cap nhat lai test cung luc voi code.

Chay: python tests/test_lang_task1.py
"""
import inspect

from app.config.schema import AppConfig, BaseConfig, DataGenV2Config, ModelsConfig, WindowConfig
from app.lang import ast_nodes as ast_nodes_module
from app.lang.parser import Parser
from app.lang.semantic import SemanticChecker
from app.lang.tokens import TokenType

results = []


def check(name: str, condition: bool) -> None:
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


# =====================================================================
# Config test-only (KHONG phai config that cua project) -- input_candles
# nho de test de doc. bin_max/digit_pad/zone_width lay tuong tu configs/base.yaml
# that nhung thu nho pham vi cho gon.
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
    action_types=("BUY", "SELL", "CANCEL_BUY", "CANCEL_SELL", "WAIT_BUY", "WAIT_SELL", "HOLD"),
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


def fmt_bin(n: int, pad: int = None) -> str:
    pad = pad if pad is not None else cfg.base.digit_pad
    return " ".join(str(n).zfill(pad))


def make_chart(closes) -> str:
    """closes: list gia Close cho tung nen, PHAI dung do dai = cfg.window.input_candles
    (tru khi co chu y test sai so luong nen)."""
    candles = [f"<O_{c}> <H_{c + 5}> <L_{c - 5}> <C_{c}>" for c in closes]
    return "<chart> " + " ".join(candles) + " </chart>"


def make_think(trend: str, current_price: int, zone=None) -> str:
    text = f"<think> <trend>{trend}</trend> <current_price> {fmt_bin(current_price)} </current_price>"
    if zone is not None:
        direction, lower, upper = zone
        tag = "zone_support" if direction == "support" else "zone_resistance"
        text += f" <{tag}> {fmt_bin(lower)} : {fmt_bin(upper)} </{tag}>"
    text += " </think>"
    return text


def semantic_check(program):
    return SemanticChecker(cfg.base.zone_width_min_bins, cfg.base.zone_width_max_bins).check(program)


# =====================================================================
# NHOM 1 -- hanh vi dung mong doi
# =====================================================================

# --- T1: RANGE, khong zone -> well-formed + semantic pass ---
closes = [500, 505, 503, 507, 510]
text = make_chart(closes) + " " + make_think("RANGE", closes[-1])
pr = Parser.from_text(cfg, text).parse()
check("T1: RANGE khong zone -> well_formed", pr.is_well_formed())
if pr.is_well_formed():
    check("T1: RANGE khong zone -> semantic passed", semantic_check(pr.ast).passed)

# --- T2: UP + zone_support hop le (duoi/chua current_price, width hop le) ---
closes = [500, 505, 503, 507, 512]
text = make_chart(closes) + " " + make_think("UP", 512, zone=("support", 500, 510))
pr = Parser.from_text(cfg, text).parse()
check("T2: UP+zone_support hop le -> well_formed", pr.is_well_formed())
if pr.is_well_formed():
    check("T2: UP+zone_support hop le -> semantic passed", semantic_check(pr.ast).passed)

# --- T3: UP nhung thieu zone -> semantic FAIL (rule A: trend<->zone) ---
closes = [500, 505, 503, 507, 512]
text = make_chart(closes) + " " + make_think("UP", 512, zone=None)
pr = Parser.from_text(cfg, text).parse()
check("T3: UP thieu zone van well_formed ve cu phap", pr.is_well_formed())
if pr.is_well_formed():
    check("T3: UP thieu zone -> semantic FAIL (rule A)", not semantic_check(pr.ast).passed)

# --- T4: zone_support nam hoan toan TREN current_price -> semantic FAIL (rule B) ---
closes = [500, 505, 503, 507, 480]
text = make_chart(closes) + " " + make_think("UP", 480, zone=("support", 500, 510))
pr = Parser.from_text(cfg, text).parse()
check("T4: well_formed (loi nay la semantic, khong phai cu phap)", pr.is_well_formed())
if pr.is_well_formed():
    check("T4: zone_support sai phia -> semantic FAIL (rule B)", not semantic_check(pr.ast).passed)

# --- T5: zone qua hep (width < zone_width_min_bins) -> semantic FAIL (rule B2) ---
closes = [500, 505, 503, 507, 512]
text = make_chart(closes) + " " + make_think("UP", 512, zone=("support", 505, 507))  # width=2 < 5
pr = Parser.from_text(cfg, text).parse()
check("T5: well_formed (Parser KHONG tu kiem tra zone width)", pr.is_well_formed())
if pr.is_well_formed():
    check("T5: zone qua hep -> semantic FAIL (rule B2)", not semantic_check(pr.ast).passed)

# --- T6: sai so luong nen trong chart -> KHONG well_formed, nhung van parse duoc (severity=value) ---
closes_short = [500, 505, 503, 507]  # thieu 1 nen so voi N_CANDLES=5
text = make_chart(closes_short) + " " + make_think("RANGE", closes_short[-1])
pr = Parser.from_text(cfg, text).parse()
check("T6: thieu nen -> KHONG well_formed", not pr.is_well_formed())
check("T6: thieu nen -> well_form_score nam trong (0,1) (van con gradient, khong ve 0)",
      0.0 < pr.well_form_score() < 1.0)

# --- T7: current_price sai so luong digit (2 thay vi 4) -> loi value, van best-effort parse ra gia tri ---
closes = [500, 505, 503, 507, 512]
text = make_chart(closes) + " <think> <trend>RANGE</trend> <current_price> 5 1 </current_price> </think>"
pr = Parser.from_text(cfg, text).parse()
check("T7: current_price thieu digit -> KHONG well_formed", not pr.is_well_formed())
check("T7: nhung van best-effort parse ra current_price_bin=51", pr.ast.think.current_price_bin == 51)

# --- T8: text du thua sau </think> -> KHONG well_formed ---
closes = [500, 505, 503, 507, 512]
text = make_chart(closes) + " " + make_think("RANGE", 512) + " <action> BUY </action>"
pr = Parser.from_text(cfg, text).parse()
check("T8: text du thua sau think_block -> KHONG well_formed", not pr.is_well_formed())
if pr.errors:
    print(f"    (T8 debug) loi dau tien: {pr.errors[0].message}")
    print("    -> neu message nhac 'ACTION_OPEN', nghia la lexer.py van con nhan dien token action "
          "thua (xem muc leftover trong review) du parser da bo han action_block.")


# --- T9: current_price SAI lech Close nen cuoi -> PHAI bi chan boi Parser (rule 2.2.C).
# DA FIX (truoc day rule nay bi xoa nham khoi parse(), gio da them lai
# _check_current_price_matches_chart() va goi trong parse()). ---
closes = [500, 505, 503, 507, 512]  # Close nen cuoi = 512
text = make_chart(closes) + " " + make_think("RANGE", 999)  # current_price=999, SAI
pr = Parser.from_text(cfg, text).parse()
check("T9: current_price sai Close nen cuoi -> KHONG well_formed (rule 2.2.C da fix)", not pr.is_well_formed())
check(
    "T9b: loi 2.2.C co severity=value (phat 0.30, nang hon structural)",
    any(e.severity == "value" and "current_price" in e.message and "Close" in e.message for e in pr.errors),
)

# --- T9c: current_price KHOP dung Close nen cuoi -> khong bi rule 2.2.C chan (control case) ---
text = make_chart(closes) + " " + make_think("RANGE", 512)  # current_price=512, DUNG
pr = Parser.from_text(cfg, text).parse()
check("T9c: current_price khop dung Close nen cuoi -> well_formed", pr.is_well_formed())

# --- T10: xac nhan ActionNode da duoc xoa khoi ast_nodes (don dep DUNG, khong phai gap) ---
check("T10: ast_nodes.py KHONG con ActionNode (don dep dung)", not hasattr(ast_nodes_module, "ActionNode"))


# =====================================================================
# NHOM 2 -- gap/leftover DA BIET. PASS o day = XAC NHAN van con ton tai
# o trang thai hien tai, KHONG phai hanh vi mong muon lau dai. Khi fix,
# cac test nay se FAIL -- do la tin hieu dung, hay xoa/sua lai luc do.
# =====================================================================

# --- T11a: ACTION_OPEN/ACTION_CLOSE/ACTION_TYPE/SL_LABEL/RR CHU Y GIU LAI --
# tokens.py la vocab dung chung giua cac project con (task2 van can action/SL/RR),
# nen day KHONG phai leftover -- chi xac nhan chung van ton tai nhu du dinh,
# khong phai loi neu sau nay ai do lo xoa nham (test se FAIL neu bi xoa).
shared_with_task2 = ["ACTION_OPEN", "ACTION_CLOSE", "ACTION_TYPE", "SL_LABEL", "RR"]
still_shared = [t for t in shared_with_task2 if hasattr(TokenType, t)]
check(
    f"T11a: {len(still_shared)}/{len(shared_with_task2)} TokenType action/SL/RR van con "
    f"(CHU Y GIU LAI cho task2 dung chung, khong phai leftover): {still_shared}",
    len(still_shared) == len(shared_with_task2),
)

# --- T11b [LEFTOVER DA BIET - NEN XOA]: PRICE_IN_ZONE/GOOD_PRICE_ACTION la 2 flag
# THAT SU chet -- task1 khong dung (ThinkNode khong co field nay), task2 (buy/hold/sell)
# cung khong can toi. Khac T11a, 2 token nay KHONG project nao con dung. ---
truly_dead_token_types = ["PRICE_IN_ZONE", "GOOD_PRICE_ACTION"]
still_present_dead = [t for t in truly_dead_token_types if hasattr(TokenType, t)]
check(
    f"T11b [LEFTOVER DA BIET - NEN XOA]: {len(still_present_dead)}/{len(truly_dead_token_types)} "
    f"TokenType chet-that-su van con: {still_present_dead}",
    len(still_present_dead) == len(truly_dead_token_types),
)

# --- T12 [MAU THUAN DOCSTRING]: SemanticChecker.__init__ chi nhan 2 tham so
# (zone_width_min_bins, zone_width_max_bins), KHONG co sl_min_dist_bins/sl_max_dist_bins
# nhu docstring class mo ta ("4 tham so ... deu bat buoc truyen qua constructor"). ---
sig = inspect.signature(SemanticChecker.__init__)
param_names = set(sig.parameters.keys()) - {"self"}
check(
    f"T12 [MAU THUAN DOCSTRING - NEN SUA]: SemanticChecker.__init__ chi nhan {sorted(param_names)}, "
    f"docstring lai noi co ca sl_min_dist_bins/sl_max_dist_bins",
    param_names == {"zone_width_min_bins", "zone_width_max_bins"},
)

print("=" * 70)
total = len(results)
passed = sum(1 for _, ok in results if ok)
print(f"TONG: {total} test, {passed} PASS, {total - passed} FAIL")
if passed != total:
    raise SystemExit(1)