friend={}

inFile = open("Brightkite_edges.txt")
line = inFile.readline()

while(line):
	data = line.split()
	if(data[0] not in friend.keys()):
		friend[data[0]] = set()
	friend[data[0]].add(data[1])
	line=inFile.readline()

suggestions = {}
person = raw_input("Enter person id : \n")
for key in friend.keys():
	if(key!= person and key not in friend[person]):
		count = len(friend[person].intersection(friend[key]))
		if(count >3):
			suggestions[key] = count

for key, value in sorted(suggestions.items(), key=lambda item: item[1],reverse=True):
	print("%s: %s" % (key, value))