import numpy as np
import matplotlib.pyplot as plt
import random

# 由题目要求知道，这里的图用邻接矩阵来表示，而不是前向星

# 这个函数是返回ER图, 我们知道邻接矩阵中，每个位置都表示一个连接状态
def CreateER(N, p):
    mat = np.random.rand(N, N)
    mat = np.where(mat>p, 0, 1)
    for i in range(N):
        mat[i, i] = 0
        mat[i, :] = mat[:, i]
    return mat

# 创建BA网络
def CreateBA(N, m, m0):
    mat = np.zeros((N, N))
    M = np.ones((m,m))
    for i in range(m):
        M[i, i] = 0
    mat[:m, :m] = M
    for i in range(m, N, 1):
        prob = [mat[:,j].sum()/mat.sum() for j in range(m)]
        selected = np.random.choice(m, m0, p=prob, replace=False)
        for k in selected:
            mat[k, i] = 1
            mat[i, k] = 1
        m = m + 1
        print("BA mat is generating %d/%d"%(m,N))
    return mat

# 画出节点的分布图,可看出结果类似正态分布
def Distribution(mat):
    (a, b) = mat.shape
    Count = np.array([mat[i, :].sum() for i in range(a)])
    hist = np.histogram(Count, bins=1000, range=(0,1000))
    plt.plot(hist[0])
    plt.xlabel('degree')
    plt.ylabel('p(degree)')
    plt.show()
    return hist

# 对一个mat进行一次SIR的传播 S 1 -- I 2 -- R 3 普通人--1 感染者--2 恢复者
def SIRSpread(mat, beta, mu, vec):
    nvec = np.array(vec)
    for i in range(vec.size):
        if vec[i] == 1:
            num = 0
            for j in range(vec.size):
                if mat[i,j] == 1 and vec[j] == 2:
                    num = num + 1
            prob = 1 - (1-beta)**num
            rand = random.random()
            if rand < prob:
                nvec[i] = 2
        elif vec[i] == 2:
            rand = random.random()
            if rand < mu:
                nvec[i] = 3
    return nvec
    

# 设置传播次数，来进行传播，并返回每个阶段S，I，R的数量
def MultiSpread(N, beta, mu, t):
    mat = CreateER(N, 0.01)
    vec = np.array([1 for i in range(N)])

    rNum = random.randint(0, N-1)
    vec[rNum] = 2

    S = []
    I = []
    R = []

    for i in range(t):
        vec = SIRSpread(mat, beta, mu, vec)
        S.append(np.where(np.array(vec)==1, 1, 0).sum())
        I.append(np.where(np.array(vec)==2, 1, 0).sum())
        R.append(np.where(np.array(vec)==3, 1, 0).sum())
    
    return S,I,R

# 画出SIR模型的统计结果
def DrawSIRResult(N, beta, mu, t):
    S,I,R = MultiSpread(N, beta, mu, t)
    X = range(t)
    fig,ax = plt.subplots()

    plt.xlabel('t')
    plt.ylabel('N(t)')

    plt.plot(X, S, marker = "o", c = "g", label="S")
    plt.plot(X, I, marker = "s", c = "r", label="I")
    plt.plot(X, R, marker = ">", c = "b", label="R")
    
    plt.legend()

    plt.show()

if __name__ == '__main__':
    DrawSIRResult(1000, 0.15, 0.3, 100)
    # BA = CreateBA(200, 3, 3)
    # Distribution(BA)
