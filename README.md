# BayesianOptimizationforExperiment
Bayesian Optimization GUI for Experiment Data

这是一个贝叶斯优化实验器，并配备了GUI界面。

若要使用GUI界面，确保安装了这些依赖库：scikit-learn, PySide2

当前目录下直接运行SweetKiss.py文件即可打开GUI界面，或者使用pyinstaller命令将其打包成.exe程序

pyinstaller -Dw SweetKiss.spec

详细的适用情形、原理和使用方法在GUI界面的“使用说明”中，这里简单介绍。

适用情形
    做实验或在生活中，你或许会遇到这样的困惑：
    因为种种原因，有天你面临一系列的选择，每个选择最终会产生不同的结果，在结果出来前，你完全不知道是好是坏。在选项很少时，你聪明的小脑袋轻松就洞悉了背后的规律，一下就做出最优的选择，又或者挨个儿尝试。但当选择里包含成百上千个选项，或选择间能自由搭配组合，甚至选项是个连续的范围时，你的困难症开始发作了。你感觉每个选择都是错的，甚至以往看起来成功的案例也不那么成功了。
    你希望做使结果最有利于你的选择，但实在条件有限，没人告诉你这些选择和结果背后的联系，你也无法同时尝试所有的选择。不仅如此，你还要为每次尝试都支付一定的代价，这种代价可能是实验耗材、测试费用、时间成本，或是某种潜在的风险。"所以你纠结于探索未知和利用已知间的平衡。于是，深夜你瘫在床上，眼一闭一睁，决定胡乱地选择，让见鬼的命运去安排这一切。
    第二天清晨，精神饱满的你，看着桌面的电脑，思绪开始发散。你想起了人工智能，继而想到梯度下降算法，又想到参数优化，想到函数的非凸性......待你厘清了思绪，感觉自己是还可以拯救一下的。
    你发现，现实或实验过程都可以看作是一个完全未知的函数，但不知其表达式、函数形式、凹凸性、导数信息等。虽然这是个“黑箱”，但作为对正态分布拥有的忠实信仰的你，你相信可以尝试尽可能少的次数，获得最有利于你的结果。
    感谢高斯贝叶斯等人在许多年前的工作，你可以用一个高斯过程来对已知样本进行回归，然后使用采样函数计算各处的采样价值并进行选择。
    对于数据，需要注意的几个点是：
    1.“GUI界面-使用说明-使用方法-图9 数据格式”中记录了准确的数据格式，可以方便地用记事本或excel打开,也可以查看“image-figure-format.jpg”
    2.已知样本条数至少应有两个。
    3.影响因素超过二十个后，优化效果非常差，这时建议你求助于神经网络等方法。
    4.取值范围需要你认真考虑后再给定，取值范围过窄可能错过好的结果，过宽则可能给你增加没有必要的工作量。
    现在可以开始你的贝叶斯优化了。
    
工作原理
    有时，我们尝试优化的代价非常高昂，甚至于无法逐一尝试，所以我们希望用较少的尝试次数，来获得较优的结果。于是，贝叶斯优化应运而生。有几种不同的方法可以执行贝叶斯优化，而最常见的是基于正态信念的贝叶斯优化（也是本优化器所执行的）。
    本优化器执行高斯过程(Gaussian Process)来创建关于目标函数分布的假设。众所周知，从高斯分布中随机抽取样本会产生一个数，那么可以简单理解为从高斯过程中抽取随机样本会产生一个函数。可以参考sklearn的说明文档https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html
    函数的形状由内核(Kernel)决定，本优化器采用的是nu=2.5的Matern内核，详见https://arxiv.org/pdf/1206.2944.pdf/
    更多内核及使用详见《内核食谱》https://www.cs.toronto.edu/~duvenaud/cookbook/
    借助高斯过程对已知样本数据进行回归，可以获得目标函数值在各处的均值和方差。如果想进一步了解如何借助高斯过程进行回归，可以阅读《机器学习的高斯过程》http://www.gaussianprocess.org/gpml/chapters/RW.pdf\"
    这里需要注意，因为实际观测的样本数据中可能存在噪声，可以在高斯过程中自定义噪声水平或从数据中学习随机误差。除了高斯过程，还有通过随机森林https://www.cs.ubc.ca/%5Csimhutter/papers/10-TR-SMAC.pdf 、Tree Parzen Estimator https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf 等建模方法，但本优化器没有用到。
    现在我们已经可以根据已有观测数据得到一个用于预测新样本的高斯过程模型，接下来我们要考虑贝叶斯优化的核心——采样函数(Acquisition Function)。采样函数的作用是让每一次采样都尽可能接近目标函数的最大值/最小值，以此提升极值点搜索效率。具体地，我们用 u(x) 表示给定高斯过程模型的采样函数，对于目标函数的下一次采样 x* = argmax u(x) 。</p>"
    常见的采样函数有PI、EI、UCB、KG、ES等及他们的组合，具体可以参考博客https://zhuanlan.zhihu.com/p/294795418 的介绍。
    这里只简单介绍EI函数与UCB函数。
    EI(Expected Improvement)通过分析采样值提升的数学期望 E[max(0, f(x)-f(x+))] 得到，公式见“image-figure-EI.jpg”
    其中 φ(·) 是标准高斯分布的概率密度函数。ξ用于平衡探索未知与利用已知，相关论文通过实验表明 ξ=0.01 可以在几乎所有实验案例中取得不错的表现。
    UCB(Upper Confidence Bound)由体现预期收益的 μ(x) 和体现风险的 κ·σ(x) 构成，并通过参数κ控制探索，公式见“image-figure-UCB.jpg”
    本优化器分别置入了q-EI方法和q-UCB方法，q指每次尽可能多地推荐采样点以供选择，获得多个推荐采样点的方式原理上与这个贝叶斯R软件包"https://github.com/AnotherSamWilson/ParBayesianOptimizatio 不谋而合，都是通过多初始点的L-BFGS-B算法获得采样函数极值点，再使用基于密度的DBSCAN方法对极值点进行聚类后平均。
    本优化器使用的都是比较稳定的方法，但不是效率最高的方法，MOE https://github.com/Yelp/MOE 提供了效率更高更稳定的q-EI算法，Cornell-MOE https://github.com/wujian16/Cornell-MOE 则在MOE的基础上开发了更多方法，比如q-KG、d-KG等，但这些方法暂时都无法迁移到本优化器上。如果是将贝叶斯优化应用于人工智能中的超参数优化，或者其他类似可计算但耗费计算资源的函数寻优，本优化器的效果并不完美，本优化器的目的仅在于设计一个GUI界面方便实验或生活中的优化。
    
文件说明：Inputtest.csv文件为示例数据文件，可通过记事本或excel打开

其他注意事项
    1. q-UCB方法暂时没有评分，适合用于简单情形，比如小范围凸优化。
    2. 对于普通办公电脑，假如超过15分钟未运行出结果，可能原因一是数据量太大，二是出现了未知错误。
    3. 一般经过五次优化后得分在5以下表明已找到最优值，或者根据学科经验或生活经验。
    
作者初学贝叶斯优化和GUI界面设计，欢迎指正。
    
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
