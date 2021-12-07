
friend={}


inFile = open("Brightkite_edges.txt")
line = inFile.readline()

while(line):
	data = line.split()
	if(data[0] not in friend.keys()):
		friend[data[0]] = set()
	friend[data[0]].add(data[1])
	line=inFile.readline()


ppl1 = raw_input("Enter person1 id : \n")
ppl2 = raw_input("Enter person2 id : \n")
mut = friend[ppl1].intersection(friend[ppl2])
print("Number of mutual friends are : " + str(len(mut))+"\n")
print(mut)
