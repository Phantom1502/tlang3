import math
# 1 cụm 5 sample trùng nhau (p=5/16), 11 sample còn lại mỗi cái unique (p=1/16 mỗi cái)
total = 25600 
probs = [7353/total] + [687/total] + [27/total] + [9922/total] + [7053/total]
h = -sum(p*math.log(p) for p in probs)
print(h)  # ≈ 2.322