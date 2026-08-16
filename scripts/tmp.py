import math
# 1 cụm 5 sample trùng nhau (p=5/16), 11 sample còn lại mỗi cái unique (p=1/16 mỗi cái)
total = 25600 
probs = [3132/total] + [11165/total] + [916/total] + [5814/total] + [3946/total]
#probs = [1/5] * 5
h = -sum(p*math.log(p) for p in probs)
print(h)  # ≈ 2.322

max_suprisal = -math.log(1.0 / 16)
print(max_suprisal)

for p in probs:
    surprisal = -math.log(p) / max_suprisal
    print(f"{p} -> {surprisal}")
    
total = 16 
probs = [2/total] * 6 + [1/total] * 4
#probs = [1/5] * 5
h = -sum(p*math.log(p) for p in probs)
print(h)  # ≈ 2.322

max_suprisal = -math.log(1.0 / 16)
print(max_suprisal)

for p in probs:
    surprisal = -math.log(p) / max_suprisal
    print(f"{p} -> {surprisal}")