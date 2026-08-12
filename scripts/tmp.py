import math
# 1 cụm 5 sample trùng nhau (p=5/16), 11 sample còn lại mỗi cái unique (p=1/16 mỗi cái)
total = 25600 
probs = [6380/total] + [8433/total] + [5647/total] + [2936/total] + [1468/total]
h = -sum(p*math.log(p) for p in probs)
print(h)  # ≈ 2.322

max_suprisal = -math.log(1.0 / 16)
print(max_suprisal)

for p in probs:
    surprisal = -math.log(p) / max_suprisal
    print(f"{p} -> {surprisal}")