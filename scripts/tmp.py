import math
# 1 cụm 5 sample trùng nhau (p=5/16), 11 sample còn lại mỗi cái unique (p=1/16 mỗi cái)
total = 16 
probs = [4/total] + [4/total] + [3/total] + [3/total] + [1/total]
h = -sum(p*math.log(p) for p in probs)
print(h)  # ≈ 2.322