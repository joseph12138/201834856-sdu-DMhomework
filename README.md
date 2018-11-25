# 201834856-sdu-DMhomework01

一、程序设计思想  
====
  （一）分词工具  
  --------
      程序采用Stanford CoreNLP作为分词工具。  
  （二）数据集的建立  
  -------
      对于每一类文件集，将前75%的数据作为训练数据，后25%的数据作为真实的测试  
      数据。在训练数据中分离出50%的文件去构造模型，用后25%的数据进行预测试。  
  （三）VSM表示与KNN参数优化    
  ----------
      利用训练数据进行向量模型的建立   
      利用训练数据中分离出来的25%的数据进行参数优化，得到最优K值  
  （四）KNN算法实现    
  ----------
      对测试数据进行KNN分类实现  
      
二、算法实现    
=====
  （一）Main函数展示    
  --------
        if __name__ == '__main__':  
          buildDict()                #扫描每一类前75%的文件建立词典，并将词典存到本地  
          openmydict()               #读取本地词典  
          trainingdata()             #用每一类前50%的数据构建模型  
          k = getfittestk()          #用每一类50%-75%的数据预测试，获取最优K值  
          cnum = test_predict(k)     #用每一类最后25%的数据进行测试，获得分类准确率  
          print("分类准确率为" + str(cnum))  
          nlp.close()  
   （二）其余功能函数展开    
   ------------
         test_predict(j): 	     ##测试数据，其中K取j维  
         getfittestk():          ##寻找最优K值  
         train_predict(j):       ##测试数据，其中K取j维  
         predicttype(d,j,t):     ##对于文件d，K的值取j，文件真实的类别为t，判断是否分类正确  
         dismin(dis,k):          ##返回数组中第k小的元素下标  
         getdis(filevec,i):      ##计算向量filevec与trainingvector中下标为i的向量之间的距离  
         trainingdata():         ##用50%的数据集建立模型  
         buildfilevector(d):     ##将当前文件表示成向量  
         openmydict():           ##将词典文件读取进来  
         def buildDict():        ##根据每个文件夹前75%的文件创建词典  
         analyzefile(d):         ##统计当前文件中的每一行，进行词频统计与记录，其中words记录单词，wordscount记录tf-idf  
         getidf(j):              ##计算单词j的idf并返回,总共用来构建词典的文件数为14121  
         haveword(d,j):          ##扫描文档d是否含有单词j，返回1代表含有，0代表不含有  
         SaveDict():             ##选前500个TF-IDF最高的,将字典内容保存为一个txt，命名为mydict  
         orderwords():           ##将词典中单词按照TF-IDF降序进行排序  
         getSum(path):           ##统计path路径中文件的个数并返回  
         
