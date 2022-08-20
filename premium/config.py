import codefast as cf

DIR = cf.io.dirname()
stop_words = set(cf.io.read(DIR + '/localdata/cn_stopwords.txt'))
