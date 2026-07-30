"""
tests/test_tokenizer.py

Test cho app/tokenizer/vocab_builder.py (+ smoke test build_tokenizer.py
neu co cai `tokenizers`). Style giong tests/testlang.py cua project (plain
assert, khong dung pytest).

Chay: python -m tests.test_tokenizer
"""
from app.tokenizer.vocab_builder import (
    ACTION_TYPES,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS_IN_ID_ORDER,
    UNK_TOKEN,
    build_vocab,
    describe_vocab,
)

results = []


def check(name: str, condition: bool) -> None:
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


# =====================================================================
# Config test-only nho gon: bin_min=0 bin_max=9 (10 gia tri/prefix candle),
# rr_min=1 rr_max=3 -- du nho de tinh tay tong so token, khong lien quan
# gi den configs/base.yaml that (giong tinh than tests/testlang.py).
# =====================================================================
BIN_MIN, BIN_MAX = 0, 9
RR_MIN, RR_MAX = 1, 3

N_CANDLE_VALUES = BIN_MAX - BIN_MIN + 1          # 10
N_RR_VALUES = RR_MAX - RR_MIN + 1                # 3
EXPECTED_TOTAL = (
    len(SPECIAL_TOKENS_IN_ID_ORDER)      # 4: unk/bos/eos/pad
    + 6                                    # structural_tags: chart/think/action open+close
    + 4 * N_CANDLE_VALUES                  # candle_O/H/L/C
    + 3                                    # trend: UP/DOWN/RANGE
    + 7                                    # digit_field_tags: current_price(2) zone_support(2)
                                            # zone_resistance(2) + "SL:"
    + 10                                   # digit 0-9
    + 1                                    # colon
    + len(ACTION_TYPES)                    # action_type (BUY/SELL/HOLD = 3)
    + N_RR_VALUES                          # rr
)

vocab = build_vocab(BIN_MIN, BIN_MAX, RR_MIN, RR_MAX)

# --- T1: tong so token dung nhu tinh tay ---
check(f"T1: build_vocab tra ve dung {EXPECTED_TOTAL} token (nhan {len(vocab)})", len(vocab) == EXPECTED_TOTAL)

# --- T2: khong id nao trung nhau ---
check("T2: khong co id trung lap giua cac nhom", len(vocab) == len(set(vocab.values())))

# --- T3: special token co id co dinh dung SPECIAL_TOKENS_IN_ID_ORDER (unk=0,bos=1,eos=2,pad=3) ---
check("T3: id dac biet co dinh unk=0/bos=1/eos=2/pad=3",
      vocab[UNK_TOKEN] == 0 and vocab[BOS_TOKEN] == 1 and vocab[EOS_TOKEN] == 2 and vocab[PAD_TOKEN] == 3)

# --- T4: idempotent -- goi lai voi cung tham so ra dung 1 ket qua ---
vocab2 = build_vocab(BIN_MIN, BIN_MAX, RR_MIN, RR_MAX)
check("T4: build_vocab idempotent (goi lai ra dung y het)", vocab == vocab2)

# --- T5: ACTION_TYPES chi con dung 3 gia tri BUY/SELL/HOLD (task2 3-action, khong con CANCEL_*/WAIT_*) ---
check("T5: ACTION_TYPES == (BUY, SELL, HOLD)", ACTION_TYPES == ("BUY", "SELL", "HOLD"))
check("T5b: vocab KHONG con token action thua (CANCEL_BUY/CANCEL_SELL/WAIT_BUY/WAIT_SELL)",
      not any(name in vocab for name in ("CANCEL_BUY", "CANCEL_SELL", "WAIT_BUY", "WAIT_SELL")))

# --- T6: 2 flag chet PRICE_IN_ZONE/GOOD_PRICE_ACTION KHONG con trong vocab (khop voi tokens.py da don) ---
check("T6: vocab KHONG con <price_in_zone>/<good_price_action>",
      "<price_in_zone>" not in vocab and "<good_price_action>" not in vocab)

# --- T7: range candle/RR trong vocab dung theo tham so truyen vao, khong bi hardcode ---
check("T7a: co token bin bien (candle_O) o hai dau range", f"<O_{BIN_MIN}>" in vocab and f"<O_{BIN_MAX}>" in vocab)
check("T7b: KHONG co token candle_O vuot ngoai range", f"<O_{BIN_MAX + 1}>" not in vocab)
check("T7c: co token RR o hai dau range", f"<RR_{RR_MIN}>" in vocab and f"<RR_{RR_MAX}>" in vocab)
check("T7d: KHONG co token RR vuot ngoai range", f"<RR_{RR_MAX + 1}>" not in vocab)

# --- T8: describe_vocab() khong crash, tong so trong bao cao khop len(vocab) ---
report = describe_vocab(BIN_MIN, BIN_MAX, RR_MIN, RR_MAX)
report_total = int(report.strip().splitlines()[-1].split("count=")[-1])
check("T8: describe_vocab() khong crash va tong khop len(vocab)", report_total == len(vocab))

# --- T9 [REGRESSION GUARD]: goi build_vocab()/describe_vocab() KHONG tham so PHAI raise TypeError.
# Day la bug da tung xay ra that (thieu args sau khi tham so hoa ham) -- neu ai vo tinh them lai
# default value cho 4 tham so nay, test se FAIL va nhac kiem tra lai co dung y khong. ---
try:
    build_vocab()
    check("T9a: build_vocab() khong tham so -> raise TypeError", False)
except TypeError:
    check("T9a: build_vocab() khong tham so -> raise TypeError", True)
except Exception as e:
    check(f"T9a: build_vocab() khong tham so -> raise TypeError (nhan nham {type(e).__name__})", False)

try:
    describe_vocab()
    check("T9b: describe_vocab() khong tham so -> raise TypeError", False)
except TypeError:
    check("T9b: describe_vocab() khong tham so -> raise TypeError", True)
except Exception as e:
    check(f"T9b: describe_vocab() khong tham so -> raise TypeError (nhan nham {type(e).__name__})", False)

# --- T10: smoke test voi gia tri GIONG configs/base.yaml that (bin_max=2047, rr_max=9) khong crash ---
try:
    prod_vocab = build_vocab(0, 2047, 1, 9)
    check("T10: build_vocab voi range that (bin_max=2047) khong crash", True)
    check("T10b: khong id trung lap voi range that", len(prod_vocab) == len(set(prod_vocab.values())))
except Exception as e:
    check(f"T10: build_vocab voi range that (bin_max=2047) khong crash (loi: {e})", False)


# =====================================================================
# Smoke test build_tokenizer.py -- CHI chay neu co cai package `tokenizers`
# (mot so may chi chuan bi data/config, chua can torch/tokenizers that).
# =====================================================================
try:
    from app.tokenizer.build_tokenizer import build_raw_tokenizer

    tok = build_raw_tokenizer(BIN_MIN, BIN_MAX, RR_MIN, RR_MAX)
    check("T11: build_raw_tokenizer khong crash, vocab_size khop build_vocab", tok.get_vocab_size() == len(vocab))

    ids = tok.encode("<chart> <O_0> <H_1> <L_2> <C_3> </chart>").ids
    decoded = tok.decode(ids)
    check("T11b: encode/decode round-trip khong lam mat token (co du <O_0>..<C_3>)",
          all(t in decoded for t in ("O_0", "H_1", "L_2", "C_3")))
except ImportError:
    print("[SKIP] T11: package `tokenizers` chua duoc cai trong moi truong nay, bo qua smoke test build_tokenizer.py")


print("=" * 70)
total = len(results)
passed = sum(1 for _, ok in results if ok)
print(f"TONG: {total} test, {passed} PASS, {total - passed} FAIL")
if passed != total:
    raise SystemExit(1)