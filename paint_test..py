import core.function as fun
import core.fileHandle as fh

Glist = fh.csvResolve('dataset/wiki-vote.csv')
fun.paint(Glist, [35,3,6,7,54,25,28,271,299], "测试")
[35, 259, 6, 7, 299, 271, 54, 55, 28]
# fun.paint(Glist,[1, 4, 6, 7, 13, 537, 425],"测试")
