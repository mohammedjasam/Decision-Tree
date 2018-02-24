sumL = []
d = {}
v = {}
t = 0

with open("data1.csv") as f:

    for l in f:
        l = l.split(",")
        if t == 0:
            for x in range(1, len(l)):
                if x == 75:
                    pass
                else:
                    d[l[x]] = x
                    v[x] = 0
            t = 1
        break

with open("data.csv") as f:

    for l in f:
        l = l.split(",")
        for x in range(1, len(l)):
            if x == 75:
                pass
            else:
                v[x] += int(l[x])

for k in d.keys():
    d[k] = v[d[k]]

sort = []

for k, v in d.items():
    sort.append((v, k))

features = []
sort = sorted(sort, reverse=True)

for i in range(100):
    features.append(sort[i][1])
print(features)
