from stanfordcorenlp import StanfordCoreNLP 
import os
nlp = StanfordCoreNLP(r'D:\000-software\corenlp\stanford-corenlp-full-2018-10-05') 

valuewords = ["NN","NNS","NNP","NNPS"]
words = []
wordscount = []
dictwords = []
trainingtags = []
trainingvector = []
path = "20news-18828"

def getSum(path):
	##BEGIN
	##统计path路径中文件的个数并返回
	allFileNum = 0
	files = os.listdir(path)
	for f in files:
		if(os.path.isfile(path+'/'+f)):
			allFileNum = allFileNum + 1
	return allFileNum
	##END


def orderwords():
	##BEGIN
	##将词典中出现的单词频率进行排序
	print("开始排序\n")
	for r1 in range(0,500):
		for r2 in range(r1+1,len(wordscount)):
			if(wordscount[r1] < wordscount[r2]):
				wordscount[r1],wordscount[r2] = wordscount[r2],wordscount[r1]
				words[r1],words[r2] = words[r2],words[r1]
	print("结束排序\n")
	##END

def SaveDict():
	##BEGIN
	##选前500个出现频率最高的,将字典内容保存为一个txt，命名为mydict
	filename = 'mydict.txt' 
	orderwords()
	len = 0
	print("开始保存\n")
	with open(filename,'a',encoding='gb18030') as p:
		for q1 in words:
			if(len<500):
				p.write(q1)
				p.write("\n")
				len = len + 1
			else:
				pass
		len = 0
		for q2 in wordscount:
			if(len<500):
				p.write(str(q2))
				p.write("\n")
				len = len + 1
			else :
				pass
	print("结束保存\n")
	##END


def analyzefile(d):
	##BEGIN
	##统计当前文件中的每一行，进行词频统计与记录，其中words记录单词，wordscount记录频率
	types = []
	try:
		f = open(d,encoding='gb18030',errors='ignore')
		for line in f.readlines(): 
			types = nlp.pos_tag(line.strip())
			for t in types:
				if(t[1] in valuewords and t[0].isalpha() and len(t[0])>2):
					if(t[0].lower() not in words):
						words.append(t[0].lower())
						wordscount.append(1)
					else:		
						wordscount[words.index(t[0].lower())] = wordscount[words.index(t[0].lower())] + 1
				else:
					pass
	finally:
		if f:
			f.close()
	##END


def buildDict():
	##BEGIN
	##根据每个文件夹前75%的文件创建词典
	t = []
	filelist = os.listdir(path)
	for fi in filelist:
		if(os.path.isdir(path+'\\'+fi)):
			t.append(path+'\\'+fi)
	for k in t:
		count = 0;
		print("正在分析"+k+'\n')
		filesum = getSum(k)
		everyfile = []
		fh = os.listdir(k)
		for m in fh:
			if(os.path.isfile(k+'\\'+m)):
				everyfile.append(k+'\\'+m)
		for d in everyfile:
			if(count < int(filesum*0.75)):
				analyzefile(d)
				count = count + 1
			else:
				pass
		print("分析完毕"+k+"\n"+"总共: "+str(count)+"个文件"+"\n")		
	SaveDict()
	print(len(words))
	#nlp.close()
	##END


def openmydict():
	##BEGIN
	##将词典文件读取进来
	try:
		g = open("mydict.txt",encoding='gb18030',errors='ignore')
		p = 0
		for k in g:
			if(p<500):
				dictwords.append(k.strip('\n'))
				p = p + 1
			else: 
				pass
	finally:
		if g:
			g.close()
	##END


def buildfilevector(d):
	##BEGIN
	##将当前文件表示成向量
	vec = []
	for i in range(0,500):
		vec.append(0)
	try:
		f = open(d,encoding='gb18030',errors='ignore')
		for line in f.readlines(): 
			types = nlp.pos_tag(line.strip())
			for t in types:
				if(t[0].lower() in dictwords):
					vec[dictwords.index(t[0].lower())] = vec[dictwords.index(t[0].lower())] + 1
				else:
					pass
	finally:
		if f:
			f.close()
	return vec
	##END



def trainingdata():
	##BEGIN
	##用50%的数据集建立模型
	t = []
	filelist = os.listdir(path)
	for fi in filelist:
		if(os.path.isdir(path+'\\'+fi)):
			t.append(path+'\\'+fi)
	typenum = 0
	for k in t:
		typenum = typenum + 1
		count = 0;
		print("正在training "+k+'\n')
		filesum = getSum(k)
		everyfile = []
		fh = os.listdir(k)
		for m in fh:
			if(os.path.isfile(k+'\\'+m)):
				everyfile.append(k+'\\'+m)
		for d in everyfile:
			if(count < int(filesum*0.5)):
				vector = buildfilevector(d)
				trainingvector.append(vector)
				trainingtags.append(str(typenum))
				count = count + 1
			else:
				pass
		print("training完毕"+k+"\n"+"总共: "+str(count)+"个文件"+"\n")
	##END


def getdis(filevec,i):
	##BEGIN
	##计算向量filevec与trainingvector中下标为i的向量之间的距离
	sum = 0
	for h in range(0,500):
		sum = sum + (int(filevec[h]) - int(trainingvector[i][h])) * (int(filevec[h]) - int(trainingvector[i][h]))
	return sum
	##END


def dismin(dis,k):
	##BEGIN
	##返回数组中第k小的元素下标
	dis1 = dis
	le = len(dis)
	for i in range(0,k):
		for j in range(i,le):
			if(dis1[i] > dis1[j]):
				mid = dis1[i]
				dis1[i] = dis1[j]
				dis1[j] = mid
	return dis.index(dis1[k-1])
	##END


def predicttype(d,j,t):
	##BEGIN
	##对于文件d，K的值取j，文件真实的类别为t，判断是否分类正确
	filevector = buildfilevector(d)
	dis = []
	for i in range(0,500):
		d = getdis(filevector,i)
		dis.append(str(d))
	t = []
	for k in range(1,j+1):
		t.append(str(trainingtags[dismin(dis,k)]))
	m = []
	for q in range(0,j):
		m.append(0)
	for q1 in range(0,j):
		for q2 in range(0,j):
			if(t[q1] == t[q2]):
				m[q1] = m[q1] + 1
	ptype = int(t[m.index(max(m))])
	if(ptype == t):
		return 1
	else:
		return 0
	##END


def train_predict(j):
	##BEGIN
	##测试数据，其中K取j维
	t = []
	filelist = os.listdir(path)
	for fi in filelist:
		if(os.path.isdir(path+'\\'+fi)):
			t.append(path+'\\'+fi)
	type5 = 0
	for k in t:
		type5 = type5 + 1
		count = 0;
		print("正在预测训练数据"+k+'\n')
		filesum = getSum(k)
		correct = 0
		everyfile = []
		fh = os.listdir(k)
		for m in fh:
			if(os.path.isfile(k+'\\'+m)):
				everyfile.append(k+'\\'+m)
		for d in everyfile:
			if(count < int(filesum*0.5)):
				count = count + 1
			else:
				if (count < int(filesum*0.75)):
					m = predicttype(d,j,type5)   #0表示错误，1表示正确
					correct = correct + m
				else:
					pass
		print("预测完毕"+k+"\n"+"总共: "+str(count)+"个文件"+"\n")
	return correct
	##END


def getfittestk():
	##BEGIN
	##寻找最优K值
	m = []
	for i in range(1,6):
		t = train_predict(i)
		m.append(str(t))
	return m.index(max(m)) + 1
	##END


def test_predict(j):
	##BEGIN
	##测试数据，其中K取j维
	t = []
	filelist = os.listdir(path)
	for fi in filelist:
		if(os.path.isdir(path+'\\'+fi)):
			t.append(path+'\\'+fi)
	type5 = 0
	testsum = 0
	for k in t:
		type5 = type5 + 1
		count = 0;
		print("正在预测测试数据"+k+'\n')
		filesum = getSum(k)
		testsum = testsum + int(filesum*0.25)
		correct = 0
		everyfile = []
		fh = os.listdir(k)
		for m in fh:
			if(os.path.isfile(k+'\\'+m)):
				everyfile.append(k+'\\'+m)
		for d in everyfile:
			if(count < int(filesum*0.75)):
				count = count + 1
			else:
				if (count < int(filesum)):
					m = predicttype(d,j,type5)   #0表示错误，1表示正确
					correct = correct + m
				else:
					pass
		print("预测完毕"+k+"\n"+"总共: "+str(count)+"个文件"+"\n")
	s = str(correct/testsum)
	return s
	##END

if __name__ == '__main__':
	buildDict()
	openmydict()
	trainingdata()
	k = getfittestk()
	cnum = test_predict(k)
	print("分类准确率为" + str(cnum))
	nlp.close()

	

