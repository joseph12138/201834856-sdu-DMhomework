from stanfordcorenlp import StanfordCoreNLP 
import os
import math
nlp = StanfordCoreNLP(r'D:\000-software\corenlp\stanford-corenlp-full-2018-10-05') #corenlp包存放路径

valuewords = ["NN","NNS","NNP","NNPS"]   #有效单词类别
words = []                               #记录候选词典
wordscount = []                          #记录候选词典里每个单词的TF-IDF
dictwords = []                           #保存词典
trainingtags = []                        #训练数据的类别-500
trainingvector = []                      #训练数据的VSM-500
avg = []                                 #训练数据每一类的均值信息
var = []                                 #训练数据每一类的方差信息
path = "20news-18828"                    #data路径


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
	##将词典中单词按照TF-IDF降序进行排序
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
	##选前500个TF-IDF最高的,将字典内容保存为一个txt，命名为mydict
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

def haveword(d,j):
	##BEGIN
	##扫描文档d是否含有单词j，返回1代表含有，0代表不含有
	try:
		f = open(d,encoding='gb18030',errors='ignore')
		for line in f.readlines(): 
			types = nlp.pos_tag(line.strip())
			for t in types:
				if(j==t[0]):
					return 1
				else:
					pass
		return 0
	finally:
		if f:
			f.close()
	##END


def getidf(j):
	##BEGIN
	##计算单词j的idf并返回,总共用来构建词典的文件数为14121
	jc = 0
	nd = 14121
	t = []
	filelist = os.listdir(path)
	for fi in filelist:
		if(os.path.isdir(path+'\\'+fi)):
			t.append(path+'\\'+fi)
	for k in t:
		count = 0
		filesum = getSum(k)
		everyfile = []
		fh = os.listdir(k)
		for m in fh:
			if(os.path.isfile(k+'\\'+m)):
				everyfile.append(k+'\\'+m)
		for d in everyfile:
			if(count < int(filesum*0.75)):
				jc = jc + haveword(d,j)
				count = count + 1
			else:
				pass
	jc = jc + 1
	return math.log10(nd/jc)

	##END

def analyzefile(d):
	##BEGIN
	##统计当前文件中的每一行，进行词频统计与记录，其中words记录单词，wordscount记录tf-idf
	types = []
	tempwords = []
	tempwordscount = []
	wordsum = 0
	try:
		f = open(d,encoding='gb18030',errors='ignore')
		for line in f.readlines(): 
			types = nlp.pos_tag(line.strip())
			wordsum = wordsum + len(types)
			for t in types:
				if(t[1] in valuewords and t[0].isalpha() and len(t[0])>2):
					if(t[0].lower() not in tempwords):
						tempwords.append(t[0].lower())
						tempwordscount.append(1)
					else:		
						tempwordscount[tempwords.index(t[0].lower())] = tempwordscount[tempwords.index(t[0].lower())] + 1
				else:
					pass
		for i in tempwordscount:
			i = i / wordsum               ##得到单词的TF
		l = 0
		for j in tempwords:
			tempwordscount[l] = tempwordscount[l] * getidf(j)        #得到单词的TF-IDF
			l = l + 1
		for s in tempwords:
			if s not in words:
				words.append(s)
				wordscount.append(tempwordscount[tempwords.index(s)])
			else:
				wordscount[words.index(s)] = ( wordscount[words.index(s)] + tempwordscount[tempwords.index(s)] ) / 2   #如果词典中已经有这个单词，则对TF-IDF取平均
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
		print("分析完毕"+k+"\n"+"总共: "+str(int(count*0.75))+"个文件"+"\n")		
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
	##用75%的数据集建立模型
	t = []
	filelist = os.listdir(path)
	for fi in filelist:
		if(os.path.isdir(path+'\\'+fi)):
			t.append(path+'\\'+fi)
	typenum = 0
	for k in t:
		typenum = typenum + 1
		count = 0;
		print("正在建立模型，当前文件夹为："+k)
		filesum = getSum(k)
		everyfile = []
		fh = os.listdir(k)
		for m in fh:
			if(os.path.isfile(k+'\\'+m)):
				everyfile.append(k+'\\'+m)
		for d in everyfile:
			if(count < int(filesum*0.75)):
				vector = buildfilevector(d)
				trainingvector.append(vector)
				trainingtags.append(str(typenum))
				count = count + 1
			else:
				pass
		print("当前文件夹完毕"+k+"\n"+"总共: "+str(int(count*0.75))+"个文件"+"\n")
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


def test_predict():
	##BEGIN
	##测试数据
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
					m = predicttype(d,type5)   #0表示错误，1表示正确
					correct = correct + m
				else:
					pass
		print("预测完毕"+k+"\n"+"总共: "+str(count)+"个文件"+"\n")
	s = str(correct/testsum)
	return s
	##END



def predicttype(d,t):
	##BEGIN
	##对于文件d，文件真实的类别为t，判断是否分类正确
	filevector = buildfilevector(d)
	num = []
	for i in range(0,20):
		num.append(1)
	for i in range(0,20):
		for p in range(0,500):
			num[i] = num[i] * (math.exp((-1)*(int(filevector[p])-int(avg[i][p]))*(int(filevector[p])-int(avg[i][p])) / (var[i][p] * var[i][p])) ) / ((var[i][p])*sqrt(6.28))
	max = num[0]
	flag = 0
	for y in range(0,20):
		if(max < num[y]):
			max = num[y]
			flag = y
	if(int(y+1) == t):
		return 1
	else:
		return 0
	##END

def caculateavg():
	##BEGIN
	##计算训练数据均值
	for t in range(0,20):
		num = []
		n = -1
		for i in range(0,500):
			num.append(0)
		for r in trainingvector:
			n = n + 1
			if(int(trainingtags[n]) == t+1):
				for k in range(0,500):
					num[k] = (num[k] + int(trainingvector[n][k])) / 2
		avg.append(num)
	##END

def caculatevar():
	##BEGIN
	##计算训练数据方差
	for t in range(0,20):
		num = []
		n = -1
		ssum = 0
		for i in range(0,500):
			num.append(0)
		for r in trainingvector:
			if(int(trainingtags[n]) == t+1):
				n = n + 1
				ssum = ssum + 1
				for k in range(0,500):
					num[k] = num[k] + (int(trainingvector[n][k])-int(avg[t][k])) * (int(trainingvector[n][k])-int(avg[t][k])) 
			else:
				n = n + 1
		for d in range(0,500):
			num[d] = num[d] / ssum
		avg.append(num)
	##END

if __name__ == '__main__':
	buildDict()               #扫描每一类前75%的文件建立词典，并将词典存到本地
	openmydict()               #读取本地词典
	trainingdata()             #用每一类前75%的数据构建模型
	caculateavg()              #计算词典每一类文件每一属性上的均值
	caculatevar()              #计算词典每一类文件每一属性上的方差
	
	cnum = test_predict()     #用每一类最后25%的数据进行测试，获得分类准确率
	print("分类准确率为" + str(cnum))
	nlp.close()

	

