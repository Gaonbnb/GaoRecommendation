# jupyter理解
对于jupyter，应该是一种编辑器的东西，然后通过端口进行接入操作，踩过的坑包括jupyter的kernel选择的是开辟的虚拟环境，但是在jupyter中写！pip install等等的终端操作都是在原生环境中进行的操作，这样就对所有的用户环境都进行了破坏
所以想对运行环境进行操作的时候一定要在终端进入虚拟环境之后然后再pip install等等。

# python语言版本
tensorflow几天装不上1.x，最后通过将python版本下降安装上了，一定要记住，血泪的教训

# 虚拟环境
可以利用python本身的venv 还有一些虚拟环境包进行创建，也可以加入有conda就利用conda构建虚拟环境，然后还可以利用docker直接虚拟一整个环境