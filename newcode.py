l = ['"subject"', '"can"', '"will"', '"one"', '"writes"', '"article"', '"like"', '"dont"', '"just"', '"know"', '"get"', '"people"', '"think"', '"also"', '"use"', '"time"', '"good"', '"well"', '"even"', '"new"', '"now"', '"way"', '"god"', '"much"', '"see"', '"first"', '"anyone"', '"many"', '"make"', '"say"', '"two"', '"may"', '"right"', '"want"', '"said"', '"really"', '"government"', '"need"', '"windows"', '"work"', '"thanks"', '"file"', '"believe"', '"system"', '"since"', '"something"', '"problem"', '"years"', '"ive"', '"game"', '"might"', '"help"', '"using"', '"used"', '"point"', '"email"', '"please"', '"space"', '"still"', '"jesus"', '"team"', '"things"', '"car"', '"drive"', '"never"', '"last"', '"take"', '"program"', '"key"', '"fact"', '"back"', '"christian"', '"going"', '"israel"', '"image"', '"made"', '"year"', '"gun"', '"must"', '"armenian"', '"state"', '"encryption"', '"games"', '"clipper"', '"window"', '"law"', '"turkish"', '"another"', '"chip"', '"come"', '"armenians"', '"bible"', '"files"', '"win"', '"jews"', '"world"', '"read"', '"sale"', '"dos"', '"players"']

index = []
with open("data1.csv") as f:
    for line in f:
        line = line.split(",")[:-1]
        for x in range(len(line)):
            if line[x] in l:
                index.append(x)
        break
# print(index)

neworder = []
header = ""
with open("data1.csv") as f:
    for line in f:
        line = line.split(",")[:-1]
        for x in range(len(line)):
            if x in index:
                header += line[x] + ","
        break

header = header[:-1]
header = header.split(",")

print(header, len(header))

with open("final.csv", "w") as f:
    print(header, file = f)

with open("final.csv", "a") as f1:
    with open("data1.csv") as f:
        for line in f:
            elements = []
            line = line.split(",")[:-1]
            for x in range(len(line)):
                if x in index:
                    elements.append(line[x])
            print(elements, file = f1)
