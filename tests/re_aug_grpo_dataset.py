import math
n = 13
probs = [n/16] + [1/16]*(16-n)
h = -sum(p*math.log(p) for p in probs)
print(h)  # ≈ 2.322