from app.config.schema import (
    BaseConfig,
)

results = []

def check(name, fn):
    try:
        fn()
        results.append((name, "FAIL (khong raise)"))
    except ValueError:
        results.append((name, "PASS (raise ValueError dung nhu ky vong)"))
    except Exception as e:  # noqa
        results.append((name, f"FAIL (raise sai loai: {type(e).__name__}: {e})"))

def check_ok(name, fn):
    print(f"TEST: {name}, {fn}")
    try:
        fn()
        results.append((name, "PASS (khoi tao thanh cong, khong raise)"))
    except Exception as e:  # noqa
        results.append((name, f"FAIL (raise ngoai y muon: {type(e).__name__}: {e})"))

if __name__ == "__main__":
    # 1. Import (da test rieng, nhung test lai trong file nay cho day du)
    check_ok(
        "Import 6 dataclass",
        lambda: (BaseConfig),
    )
    
    check_ok(
        "BaseConfig khoi tao voi du field",
        lambda: BaseConfig(
            bin_min=0,
            bin_max=2047,
            digit_pad=4,
            rr_min=1,
            rr_max=9,
            action_types=("BUY", "SELL", "CANCEL_BUY", "CANCEL_SELL", "WAIT_BUY", "WAIT_SELL", "HOLD"),
            trend_values=("UP", "DOWN", "RANGE"),
        ),
    )


    print("=" * 70)
    n_fail = 0
    for name, status in results:
        print(f"[{status.split()[0]:4}] {name} -> {status}")
        if status.startswith("FAIL"):
            n_fail += 1
    print("=" * 70)
    print(f"TONG: {len(results)} test, {len(results) - n_fail} PASS, {n_fail} FAIL")
    if n_fail:
        raise SystemExit(1)