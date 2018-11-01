from stanfordcorenlp import StanfordCoreNLP 
import os
nlp = StanfordCoreNLP(r'D:\000-software\corenlp\stanford-corenlp-full-2018-10-05') 

#valuewords = ["NN","NNS","NNP","NNPS","VB","VBG","VBD","VBN","VBP","VBZ"]
valuewords = ["NN","NNS","NNP","NNPS"]
stops = ["``","''",",",".","POS","PRP","PRP$","RB","RBR","RBS","WDT","WP","WP$","WRB","MD","JJ","JJR","JJS","SYM","TO","UH","IN","-LRB-","-RRB-",":"]
words = []
wordscount = []
dictwords = []
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
	for r1 in range(0,200):
		for r2 in range(r1+1,len(wordscount)):
			if(wordscount[r1] < wordscount[r2]):
				wordscount[r1],wordscount[r2] = wordscount[r2],wordscount[r1]
				words[r1],words[r2] = words[r2],words[r1]
	print("结束排序\n")
	##END

def SaveDict():
	##BEGIN
	##选前100个出现频率最高的,将字典内容保存为一个txt，命名为mydict
	filename = 'mydict.txt' 
	orderwords()
	len = 0
	print("开始保存\n")
	with open(filename,'a',encoding='gb18030') as p:
		for q1 in words:
			if(len<200):
				p.write(q1)
				p.write("\n")
				len = len + 1
			else:
				pass
		len = 0
		for q2 in wordscount:
			if(len<200):
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
	nlp.close()
	##END


def openmydict():
	##BEGIN
	##将词典文件读取进来
	try:
		g = open("mydict.txt",encoding='gb18030',errors='ignore')
		p = 0
		for k in g:
			if(p<200):
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
	for i in range(0,200):
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



trainingtags = []
trainingvector = []
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
				trainingvector.append(str(vector))
				trainingtags.append(str(typenum))
				count = count + 1
			else:
				pass
		print("training完毕"+k+"\n"+"总共: "+str(count)+"个文件"+"\n")
	##END


if __name__ == '__main__':
	#buildDict()
	openmydict()
	trainingdata()
	

	print(trainingvector)
	print(trainingtags)
	print(len(trainingtags))
	print("\n")
	print(len(trainingvector))
	print("\n")

