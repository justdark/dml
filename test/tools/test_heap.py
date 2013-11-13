from dml.tool import Heap
def cmp(aa,bb):
	return aa<bb
a=Heap()
a.insert(36)
a.insert(45)

a.insert(72)
a.insert(30)
a.insert(18)
a.insert(53)
a.insert(35)
a.insert(48)
a.insert(93)
a.insert(15)

print a.heap
while a.counter>0:
	print a.heap[1]
	#print a.heap
	a.delete(1)