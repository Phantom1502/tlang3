import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

from app.config.loader import load_config, get_scale, get_round_config
from app.config.schema import AppConfig

FIXTURE_DIR = os.path.join("configs")
TMP_DIR = os.path.join(os.path.dirname(__file__), "_tmp_config_variant")

print("=" * 70)
print(f"TEST 3: load_config() from {FIXTURE_DIR}")

results = []


def check(name, condition):
    results.append((name, condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


# 1) load_config voi fixture hop le -> tra ve dung 1 AppConfig, khong loi
cfg = load_config(FIXTURE_DIR)
check("load_config(fixture hop le) tra ve AppConfig", isinstance(cfg, AppConfig))

# 2) xoa 1 file bat buoc (window.yaml) -> load_config() raise FileNotFoundError
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
shutil.copytree(FIXTURE_DIR, TMP_DIR)
os.remove(os.path.join(TMP_DIR, "window.yaml"))
try:
    load_config(TMP_DIR)
    check("xoa window.yaml -> raise FileNotFoundError", False)
except FileNotFoundError:
    check("xoa window.yaml -> raise FileNotFoundError", True)
except Exception as e:
    check(f"xoa window.yaml -> raise FileNotFoundError (got {type(e).__name__} instead)", False)
shutil.rmtree(TMP_DIR)

# 3) sua 1 field bat buoc thanh thieu trong base.yaml -> raise ValueError
shutil.copytree(FIXTURE_DIR, TMP_DIR)
base_path = os.path.join(TMP_DIR, "base.yaml")
with open(base_path, "r") as f:
    content = f.read()
# xoa dong rr_max de tao thieu field bat buoc
new_content = "\n".join(
    line for line in content.splitlines() if not line.startswith("rr_max:")
)
with open(base_path, "w") as f:
    f.write(new_content)
try:
    load_config(TMP_DIR)
    check("thieu rr_max trong base.yaml -> raise ValueError", False)
except ValueError:
    check("thieu rr_max trong base.yaml -> raise ValueError", True)
except Exception as e:
    check(f"thieu rr_max trong base.yaml -> raise ValueError (got {type(e).__name__} instead)", False)
shutil.rmtree(TMP_DIR)

# 4) get_scale(cfg, "XAUUSD", "M1", 100) tra dung float da ghi trong fixture
scale_val = get_scale(cfg, "XAUUSD", "M1", 100)
check(f"get_scale(XAUUSD,M1,100) == 24.0 (got {scale_val})", scale_val == 24.0)

# 5) get_scale(cfg, "XAUUSD", "M1", 999) (window_size khong ton tai) raise KeyError
try:
    get_scale(cfg, "XAUUSD", "M1", 999)
    check("get_scale window_size=999 -> raise KeyError", False)
except KeyError:
    check("get_scale window_size=999 -> raise KeyError", True)
except Exception as e:
    check(f"get_scale window_size=999 -> raise KeyError (got {type(e).__name__} instead)", False)

# Bo sung: get_round_config
rc = get_round_config(cfg, "round1")
check("get_round_config(round1) tra dung RoundConfig", rc.round_id == "round1")
try:
    get_round_config(cfg, "round_khong_ton_tai")
    check("get_round_config round khong ton tai -> raise KeyError", False)
except KeyError:
    check("get_round_config round khong ton tai -> raise KeyError", True)

print("=" * 70)
total = len(results)
passed = sum(1 for _, ok in results if ok)
print(f"TONG: {total} test, {passed} PASS, {total - passed} FAIL")
if passed != total:
    sys.exit(1)