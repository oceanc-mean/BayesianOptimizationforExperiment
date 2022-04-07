# coding=utf-8
# @Time : 2022/2/27 17:18
# @Author : Ohmic Lab
# @File : SweetKiss.py 
# @Software: PyCharm

import os
import sys
import base64
import inspect
import ctypes
import threading
import numpy as np
from qss import qss
from functools import partial
from BayesianOptimization import BO
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from PySide2.QtCore import (
    QStandardPaths,
    QDir,
    Qt
)
from PySide2.QtGui import (
    QIcon,
    QStandardItemModel,
    QStandardItem,
    QColor,
    QFont,
    QPainter,
    QPen,
    QBrush,
    QPixmap,
    QCursor
)
from PySide2.QtWidgets import (
    QAction,
    QWidget,
    QMainWindow,
    QDialog,
    QApplication,
    QFormLayout,
    QComboBox,
    QLabel,
    QRadioButton,
    QTableView,
    QLineEdit,
    QScrollArea,
    QDoubleSpinBox,
    QProgressBar,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QStyle,
    QFileDialog,
    QToolTip
)
from PySide2.QtCharts import QtCharts
from figure import Figure

#  pyinstaller -Dw SweetKiss.spec
def main():
    # Parameters

    # Functions
    app = QApplication(sys.argv)
    with open('tmp.ico', 'wb') as tmp:
        tmp.write(base64.b64decode(Figure().img))
    app.setWindowIcon(QIcon("tmp.ico"))
    os.remove('tmp.ico')

    widget = Widget()
    main_window = MainWindow(widget)
    main_window.show()

    _style = qss
    app.setStyleSheet(_style)

    sys.exit(app.exec_())


class MainWindow(QMainWindow):
    def __init__(self, widget):
        super(MainWindow, self).__init__()
        # 设置主窗口标题
        self.setWindowTitle("贝叶斯实验优化器")
        self.setCentralWidget(widget)

        # 设置菜单栏
        bar = self.menuBar()
        instructions = bar.addMenu("使用说明")

        applicable_situation = QAction("适用情形", self)
        applicable_situation.setShortcut(Qt.ALT | Qt.Key_A)
        applicable_situation.triggered.connect(self.situation)
        instructions.addAction(applicable_situation)

        working_principle = QAction("工作原理", self)
        working_principle.setShortcut(Qt.ALT | Qt.Key_S)
        working_principle.triggered.connect(self.principle)
        instructions.addAction(working_principle)

        using_method = QAction("使用方法", self)
        using_method.setShortcut(Qt.ALT | Qt.Key_D)
        using_method.triggered.connect(self.method)
        instructions.addAction(using_method)

        # 设置主窗口绝对尺寸
        self.resize(572, 361)
        self.setMaximumSize(572, 361)
        # self.setFixedSize(self.width(), self.height())

    @staticmethod
    def situation():
        tip = QDialog()
        tip.setWindowTitle("适用情形")

        title = QLabel()
        title.setText("<p>适用情形</p>")
        title.setStyleSheet("font-family: 宋体; font: bold; min-width: 10em; font-size: 22px;")

        para1 = QLabel()
        para1.setText('<p style="text-indent:36px;">做实验或在生活中，你或许会遇到这样的困惑：</p>'
                      '<p style="text-indent:36px;">因为种种原因，有天你面临一系列的选择，每个选择最终会产生不同的结果，在结果出来前，你完全不知道是好是坏。'
                      "在选项很少时，你聪明的小脑袋轻松就洞悉了背后的规律，一下就做出最优的选择，又或者挨个儿尝试。"
                      "但当选择里包含成百上千个选项，或选择间能自由搭配组合，甚至选项是个连续的范围时，你的困难症开始发作了。"
                      '你感觉每个选择都是错的，甚至以往看起来成功的案例也不那么成功了。</p>'
                      '<p style="text-indent:36px;">你希望做使结果最有利于你的选择，但实在条件有限，没人告诉你这些选择和结果背后的联系，你也无法同时尝试所有的选择。'
                      "不仅如此，你还要为每次尝试都支付一定的代价，这种代价可能是实验耗材、测试费用、时间成本，或是某种潜在的风险。"
                      "所以你纠结于探索未知和利用已知间的平衡。"
                      '于是，深夜你瘫在床上，眼一闭一睁，决定胡乱地选择，让见鬼的命运去安排这一切。</p>'
                      '<p style="text-indent:36px;">第二天清晨，精神饱满的你，看着桌面的电脑，思绪开始发散。'
                      "你想起了人工智能，继而想到梯度下降算法，又想到参数优化，想到函数的非凸性......"
                      '待你厘清了思绪，感觉自己是还可以拯救一下的。</p>')
        para1.setWordWrap(True)
        para1.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image1 = QLabel()
        with open('tmp1.jpeg', 'wb') as tmp1:
            tmp1.write(base64.b64decode(Figure().img1))
        image1.setPixmap(QPixmap("tmp1.jpeg"))
        image1.setFixedSize(200, 200)
        image1.setScaledContents(True)
        os.remove('tmp1.jpeg')

        para2 = QLabel()
        para2.setText('<p style="text-indent:36px;">你发现，现实或实验过程都可以看作是一个完全未知的函数，但不知其表达式、函数形式、凹凸性、导数信息等。'
                      '虽然这是个“黑箱”，但作为对正态分布拥有的忠实信仰的你，你相信可以尝试尽可能少的次数，获得最有利于你的结果。</p>'
                      '<p style="text-indent:36px;">感谢高斯贝叶斯等人在许多年前的工作，你可以用一个高斯过程来对已知样本进行回归，'
                      '然后使用采样函数计算各处的采样价值并进行选择。</p>'
                      '<p style="text-indent:36px;">对于数据，需要注意的几个点是：</p>'
                      '<p style="text-indent:36px;">1.“使用说明-使用方法-图9 数据格式”中记录了准确的数据格式，可以方便地用记事本或excel打开。</p>'
                      '<p style="text-indent:36px;">2.已知样本条数至少应有两个。</p>'
                      '<p style="text-indent:36px;">3.影响因素超过二十个后，优化效果非常差，这时建议你求助于神经网络等方法。</p>'
                      '<p style="text-indent:36px;">4.取值范围需要你认真考虑后再给定，取值范围过窄可能错过好的结果，过宽则可能给你增加没有必要的工作量。</p><br>'
                      '<p style="text-indent:36px;">现在可以开始你的贝叶斯优化了。</p><br>')
        para2.setWordWrap(True)
        para2.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        layout = QVBoxLayout()
        layout.addWidget(title, 1, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(para1, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image1, 1, Qt.AlignCenter)
        layout.addWidget(para2, Qt.AlignLeft | Qt.AlignTop)

        widget = QWidget()
        widget.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: 0px; border-radius: 0px;")

        lay_out = QVBoxLayout()
        lay_out.addWidget(scroll)
        tip.setLayout(lay_out)
        tip.setMinimumSize(858, 361*2)
        tip.exec_()

    @staticmethod
    def principle():
        tip = QDialog()
        tip.setWindowTitle("工作原理")

        title = QLabel()
        title.setText("<p>工作原理</p>")
        title.setStyleSheet("font-family: 宋体; font: bold; min-width: 10em; font-size: 22px;")

        para1 = QLabel()
        para1.setText(
            '<p style="text-indent:36px;">有时，我们尝试优化的代价非常高昂，甚至于无法逐一尝试，所以我们希望用较少的尝试次数，来获得较优的结果。'
            "于是，贝叶斯优化应运而生。</p>"
            '<p style="text-indent:36px;">有几种不同的方法可以执行贝叶斯优化，而最常见的是基于正态信念的贝叶斯优化（也是本优化器所执行的）。'
            "本优化器执行高斯过程(Gaussian Process)来创建关于目标函数分布的假设。"
            "众所周知，从高斯分布中随机抽取样本会产生一个数，"
            "那么可以简单理解为从高斯过程中抽取随机样本会产生一个函数（如图1）。</p>")
        para1.setWordWrap(True)
        para1.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image1 = QLabel()
        with open('tmp2.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img2))
        image1.setPixmap(QPixmap("tmp2.jpg"))
        image1.setFixedSize(465, 210)
        image1.setScaledContents(True)
        os.remove('tmp2.jpg')

        annotate1 = QLabel()
        annotate1.setText("<p>图1 高斯过程抽样</p>")
        annotate1.setWordWrap(True)
        annotate1.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        annotate1_ect = QLabel()
        annotate1_ect.setText(
            '<a style="text-decoration:none;color:black;" '
            'href= \"https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html\"> '
            '图片来自scikit-learn说明文档(Gaussian Process Regression) </a>')
        annotate1_ect.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 14px;")
        annotate1_ect.setOpenExternalLinks(True)

        para2 = QLabel()
        para2.setText(
            '<p style="text-indent:36px;">函数的形状由内核(Kernel)决定，'
            '本优化器采用的是nu=2.5的Matern内核'
            '<sup>[<a style="text-decoration:none;" '
            'href= \"https://arxiv.org/pdf/1206.2944.pdf/\" >1</a>]</sup>'
            '，'
            '更多内核及使用详见《内核食谱》'
            '<sup>[<a style="text-decoration:none;" '
            'href= \"https://www.cs.toronto.edu/~duvenaud/cookbook/\" >2</a>]</sup>'
            '。</p>'
            '<p style="text-indent:36px;">借助高斯过程对已知样本数据进行回归，可以获得目标函数值在各处的均值和方差。'
            "如果想进一步了解如何借助高斯过程进行回归，可以阅读《机器学习的高斯过程》"
            '<sup>[<a style="text-decoration:none;" '
            'href= \"http://www.gaussianprocess.org/gpml/chapters/RW.pdf\" >3</a>]</sup>'
            "。"
            "这里需要注意，因为实际观测的样本数据中可能存在噪声，可以在高斯过程中自定义噪声水平或从数据中学习随机误差。"
            "除了高斯过程，还有通过随机森林"
            '<sup>[<a style="text-decoration:none;" '
            'href= \"https://www.cs.ubc.ca/%5Csimhutter/papers/10-TR-SMAC.pdf\" >4</a>]</sup>'
            "、Tree Parzen Estimator"
            '<sup>[<a style="text-decoration:none;" '
            'href= \"https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf\" >5</a>]</sup>'
            "等建模方法，但本优化器没有用到。</p>"
            '<p style="text-indent:36px;">现在我们已经可以根据已有观测数据得到一个用于预测新样本的高斯过程模型，'
            "接下来我们要考虑贝叶斯优化的核心——采样函数(Acquisition Function)。"
            "采样函数的作用是让每一次采样都尽可能接近目标函数的最大值/最小值，以此提升极值点搜索效率。"
            "具体地，我们用 u(x) 表示给定高斯过程模型的采样函数，对于目标函数的下一次采样 x* = argmax u(x) 。</p>"
            '<p style="text-indent:36px;">常见的采样函数有PI、EI、UCB、KG、ES等及他们的组合，具体可以参考博客'
            '<sup>[<a style="text-decoration:none;" '
            'href= \"https://zhuanlan.zhihu.com/p/294795418\" >6</a>]</sup>'
            '的介绍。'
            "这里只简单介绍EI函数与UCB函数。</p>"
            '<p style="text-indent:36px;">EI(Expected Improvement)通过分析采样值提升的数学期望 E[max(0, f(x)-f(x+))] 得到，公式如下：</p>')
        para2.setWordWrap(True)
        para2.setOpenExternalLinks(True)
        para2.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image2 = QLabel()
        with open('tmp3.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img3))
        image2.setPixmap(QPixmap('tmp3.jpg'))
        image2.setFixedSize(557, 157)
        image2.setScaledContents(True)
        os.remove('tmp3.jpg')

        annotate2 = QLabel()
        annotate2.setText("<p>图2 EI采样公式</p>")
        annotate2.setWordWrap(True)
        annotate2.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        para3 = QLabel()
        para3.setText(
            '<p style="text-indent:36px;">其中 φ(·) 是标准高斯分布的概率密度函数。'
            'ξ用于平衡探索未知与利用已知，相关论文通过实验表明 ξ=0.01 可以在几乎所有实验案例中取得不错的表现。</p>'
            '<p style="text-indent:36px;">UCB(Upper Confidence Bound)由体现预期收益的 μ(x) 和体现风险的 κ·σ(x) 构成，'
            '并通过参数κ控制探索，公式如下：</p>')
        para3.setWordWrap(True)
        para3.setOpenExternalLinks(True)
        para3.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image3 = QLabel()
        with open('tmp4.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img4))
        image3.setPixmap(QPixmap("tmp4.jpg"))
        image3.setFixedSize(314, 29)
        image3.setScaledContents(True)
        os.remove('tmp4.jpg')

        annotate3 = QLabel()
        annotate3.setText("<p>图3 UCB采样公式</p>")
        annotate3.setWordWrap(True)
        annotate3.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        para4 = QLabel()
        para4.setText(
            '<p style="text-indent:36px;">本优化器分别置入了q-EI方法和q-UCB方法，q指每次尽可能多地推荐采样点以供选择，'
            "获得多个推荐采样点的方式原理上与这个贝叶斯R软件包"
            '<sup>[<a style="text-decoration:none;" '
            'href= \"https://github.com/AnotherSamWilson/ParBayesianOptimization\" >7</a>]</sup>'
            "不谋而合，"
            "都是通过多初始点的L-BFGS-B算法获得采样函数极值点，再使用基于密度的DBSCAN方法对极值点进行聚类后平均。</p>"
            '<p style="text-indent:36px;">本优化器使用的都是比较稳定的方法，但不是效率最高的方法，MOE'
            '<sup>[<a style="text-decoration:none;" '
            'href= \"https://github.com/Yelp/MOE\" >8</a>]</sup>'
            '提供了效率更高更稳定的q-EI算法，'
            "Cornell-MOE"
            '<sup>[<a style="text-decoration:none;" '
            'href= \"https://github.com/wujian16/Cornell-MOE\" >9</a>]</sup>'
            "则在MOE的基础上开发了更多方法，比如q-KG、d-KG等，但这些方法暂时都无法迁移到本优化器上。"
            "如果是将贝叶斯优化应用于人工智能中的超参数优化，或者其他类似可计算但耗费计算资源的函数寻优，"
            "本优化器的效果并不完美，本优化器的目的仅在于实验或生活中的优化。</p>"
            "")
        para4.setWordWrap(True)
        para4.setOpenExternalLinks(True)
        para4.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        reference_title = QLabel()
        reference_title.setText("\n参考资料")
        reference_title.setStyleSheet("font-family: 宋体; font: bold; min-width: 10em; font-size: 20px;")

        reference1 = QLabel()
        reference1.setText("<a href= \"https://arxiv.org/pdf/1206.2944.pdf/\" >"
                           "[1] Practical Bayesian Optimization of Machine Learning Algorithms. "
                           "Jasper Snoek, Hugo Larochelle, Ryan P. Adams. 2012.</a>")
        reference1.setOpenExternalLinks(True)
        reference1.setWordWrap(True)
        reference1.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        reference2 = QLabel()
        reference2.setText("<a href= \"https://www.cs.toronto.edu/~duvenaud/cookbook/\" >"
                           "[2] The Kernel Cookbook: Advice on Covariance functions</a>")
        reference2.setOpenExternalLinks(True)
        reference2.setWordWrap(True)
        reference2.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        reference3 = QLabel()
        reference3.setText("<a href= \"http://www.gaussianprocess.org/gpml/chapters/RW.pdf\" >"
                           "[3] Gaussian Processes for Machine Learning. C. E. Rasmussen, C. K. I. Williams. 2006.</a>")
        reference3.setOpenExternalLinks(True)
        reference3.setWordWrap(True)
        reference3.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        reference4 = QLabel()
        reference4.setText("<a href= \"https://www.cs.ubc.ca/%5Csimhutter/papers/10-TR-SMAC.pdf\" >"
                           "[4] Sequential Model-Based Optimization for General Algorithm Configuration. "
                           "Frank Hutter, Holger H. Hoos, Kevin Leyton-Brown. 2011.</a>")
        reference4.setOpenExternalLinks(True)
        reference4.setWordWrap(True)
        reference4.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        reference5 = QLabel()
        reference5.setText(
            "<a href= \"https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf\" >"
            "[5] Algorithms for Hyper-Parameter Optimization. "
            "James Bergstra, R ́emi Bardenet, Yoshua Bengio, Bal ́azs K ́egl. 2011.</a>")
        reference5.setOpenExternalLinks(True)
        reference5.setWordWrap(True)
        reference5.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        reference6 = QLabel()
        reference6.setText(
            "<a href= \"https://zhuanlan.zhihu.com/p/294795418\" >"
            "[6] 超参数调优总结，贝叶斯优化Python代码示例.</a>")
        reference6.setOpenExternalLinks(True)
        reference6.setWordWrap(True)
        reference6.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        reference7 = QLabel()
        reference7.setText(
            "<a href= \"https://github.com/AnotherSamWilson/ParBayesianOptimization\" >"
            "[7] https://github.com/AnotherSamWilson/ParBayesianOptimization</a>")
        reference7.setOpenExternalLinks(True)
        reference7.setWordWrap(True)
        reference7.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        reference8 = QLabel()
        reference8.setText(
            "<a href= \"https://github.com/Yelp/MOE\" >"
            "[8] https://github.com/Yelp/MOE</a>")
        reference8.setOpenExternalLinks(True)
        reference8.setWordWrap(True)
        reference8.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        reference9 = QLabel()
        reference9.setText(
            "<a href= \"https://github.com/wujian16/Cornell-MOE\" >"
            "[9] https://github.com/wujian16/Cornell-MOE</a>")
        reference9.setOpenExternalLinks(True)
        reference9.setWordWrap(True)
        reference9.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        layout = QVBoxLayout()
        layout.addWidget(title, 1, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(para1, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image1, 1, Qt.AlignCenter)
        layout.addWidget(annotate1_ect, 1, Qt.AlignCenter)
        layout.addWidget(annotate1, 1, Qt.AlignCenter)
        layout.addWidget(para2, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image2, 1, Qt.AlignCenter)
        layout.addWidget(annotate2, 1, Qt.AlignCenter)
        layout.addWidget(para3, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image3, 1, Qt.AlignCenter)
        layout.addWidget(annotate3, 1, Qt.AlignCenter)
        layout.addWidget(para4, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(reference_title, 1, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(reference1, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(reference2, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(reference3, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(reference4, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(reference5, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(reference6, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(reference7, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(reference8, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(reference9, Qt.AlignLeft | Qt.AlignTop)

        widget = QWidget()
        widget.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: 0px; border-radius: 0px;")

        lay_out = QVBoxLayout()
        lay_out.addWidget(scroll)
        tip.setLayout(lay_out)
        tip.setMinimumSize(858, 361*2)
        tip.exec_()

    @staticmethod
    def method():
        tip = QDialog()
        tip.setWindowTitle("使用方法")

        title = QLabel()
        title.setText("使用方法")
        title.setStyleSheet("font-family: 宋体; font: bold; min-width: 10em; font-size: 22px;")

        para1 = QLabel()
        para1.setText(
            '<p style="text-indent:36px;">文件说明：共包含2个文件，其中.csv文件为示例数据文件，可通过记事本或excel打开，'
            '另一个.exe文件为应用文件，双击可打开本优化器。</p>'
            '<p style="text-indent:36px;">双击.exe文件打开交互界面，交互界面如下：</p>')
        para1.setWordWrap(True)
        para1.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image1 = QLabel()
        with open('tmp5.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img5))
        image1.setPixmap(QPixmap("tmp5.jpg"))
        image1.setFixedSize(572, 393)
        image1.setScaledContents(True)
        os.remove("tmp5.jpg")

        annotate1 = QLabel()
        annotate1.setText("<p>图1 初始界面</p>")
        annotate1.setWordWrap(True)
        annotate1.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        para2 = QLabel()
        para2.setText(
            '<p style="text-indent:36px;">设置参数和相关路径，鼠标悬停可触发提示信息，如下：</p>')
        para2.setWordWrap(True)
        para2.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image2 = QLabel()
        with open('tmp6.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img6))
        image2.setPixmap(QPixmap('tmp6.jpg'))
        image2.setFixedSize(572, 386)
        image2.setScaledContents(True)
        os.remove('tmp6.jpg')

        annotate2 = QLabel()
        annotate2.setText("<p>图2 提示工具</p>")
        annotate2.setWordWrap(True)
        annotate2.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        para3 = QLabel()
        para3.setText(
            '<p style="text-indent:36px;">设置好参数后点击“确认并开始运行”，即开始进行贝叶斯优化相关计算，点取消可打断计算进程，如图：</p>')
        para3.setWordWrap(True)
        para3.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image3 = QLabel()
        with open('tmp7.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img7))
        image3.setPixmap(QPixmap('tmp7.jpg'))
        image3.setFixedSize(563, 388)
        image3.setScaledContents(True)
        os.remove('tmp7.jpg')

        annotate3 = QLabel()
        annotate3.setText("<p>图3 计算进程</p>")
        annotate3.setWordWrap(True)
        annotate3.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        para4 = QLabel()
        para4.setText(
            '<p style="text-indent:36px;">当样本量或指标较多时，计算速度较慢，计算成功后主页面变化为如下：</p>')
        para4.setWordWrap(True)
        para4.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image4 = QLabel()
        with open('tmp8.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img8))
        image4.setPixmap(QPixmap('tmp8.jpg'))
        image4.setFixedSize(570, 391)
        image4.setScaledContents(True)
        os.remove('tmp8.jpg')

        annotate4 = QLabel()
        annotate4.setText("<p>图4 计算完成</p>")
        annotate4.setWordWrap(True)
        annotate4.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        para5 = QLabel()
        para5.setText(
            '<p style="text-indent:36px;">如果自动学习数据误差，建议多计算几次并保留每次的输出结果，可点击“查看优化建议”-“一键输出”，如图：</p>')
        para5.setWordWrap(True)
        para5.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image5 = QLabel()
        with open('tmp9.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img9))
        image5.setPixmap(QPixmap('tmp9.jpg'))
        image5.setFixedSize(638, 507)
        image5.setScaledContents(True)
        os.remove('tmp9.jpg')

        annotate5 = QLabel()
        annotate5.setText("<p>图5 输出结果</p>")
        annotate5.setWordWrap(True)
        annotate5.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        para6 = QLabel()
        para6.setText(
            '<p style="text-indent:36px;">本优化器可以可视化数据及优化建议，点击“优化可视化”后，弹出选择框如图，如图：</p>')
        para6.setWordWrap(True)
        para6.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image6 = QLabel()
        with open('tmp10.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img10))
        image6.setPixmap(QPixmap('tmp10.jpg'))
        image6.setFixedSize(570, 390)
        image6.setScaledContents(True)
        os.remove('tmp10.jpg')

        annotate6 = QLabel()
        annotate6.setText("<p>图6 选择对象</p>")
        annotate6.setWordWrap(True)
        annotate6.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        para7 = QLabel()
        para7.setText(
            '<p style="text-indent:36px;">选择左侧输出多因素共同作用下的可视化结果，右侧为单一因素影响下的结果，如图：</p>')
        para7.setWordWrap(True)
        para7.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image7 = QLabel()
        with open('tmp11.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img11))
        image7.setPixmap(QPixmap('tmp11.jpg'))
        image7.setFixedSize(899, 979)
        image7.setScaledContents(True)
        os.remove('tmp11.jpg')

        annotate7 = QLabel()
        annotate7.setText("<p>图7 多因素</p>")
        annotate7.setWordWrap(True)
        annotate7.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        image8 = QLabel()
        with open('tmp12.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img12))
        image8.setPixmap(QPixmap('tmp12.jpg'))
        image8.setFixedSize(888, 973)
        image8.setScaledContents(True)
        os.remove('tmp12.jpg')

        annotate8 = QLabel()
        annotate8.setText("<p>图8 单因素</p>")
        annotate8.setWordWrap(True)
        annotate8.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        para9 = QLabel()
        para9.setText(
            '<p style="text-indent:36px;">查看结果后可修改参数，重新运行计算。'
            '数据样本(.csv)的格式如下：</p>')
        para9.setWordWrap(True)
        para9.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        image9 = QLabel()
        with open('tmp13.jpg', 'wb') as tmp:
            tmp.write(base64.b64decode(Figure().img13))
        image9.setPixmap(QPixmap('tmp13.jpg'))
        image9.setFixedSize(468, 157)
        image9.setScaledContents(True)
        os.remove('tmp13.jpg')

        annotate9 = QLabel()
        annotate9.setText("<p>图9 数据格式</p>")
        annotate9.setWordWrap(True)
        annotate9.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 16px;")

        para10 = QLabel()
        para10.setText(
            '<p style="text-indent:36px;">寻优区间[Searching Start, Searching End]可以任意选定，甚至可以在已有的数据样本之外，'
            '但寻优起点必须小于寻优终点，且合理的区间宽度可减少不必要的工作量。'
            'q-UCB方法暂时没有评分，适合用于简单情形，比如小范围凸优化，</p>')
        para10.setWordWrap(True)
        para10.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        title2 = QLabel()
        title2.setText(
            '<br><p>其他注意事项</p>')
        title2.setStyleSheet("font-family: 宋体; font: bold; min-width: 10em; font-size: 22px;")

        para11 = QLabel()
        para11.setText(
            '<p style="text-indent:36px;">1. q-UCB方法暂时没有评分，适合用于简单情形，比如小范围凸优化。</p>'
            '<p style="text-indent:36px;">2. 对于普通办公电脑，假如超过15分钟未运行出结果，可能原因一是数据量太大，二是出现了未知错误。</p>'
            '<p style="text-indent:36px;">3. 一般经过五次优化后得分在5以下表明已找到最优值，或者根据学科经验或生活经验。</p>'
        )
        para11.setWordWrap(True)
        para11.setStyleSheet("font-family: 仿宋; min-width: 10em; font-size: 18px;")

        layout = QVBoxLayout()
        layout.addWidget(title, 1, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(para1, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image1, 1, Qt.AlignCenter)
        layout.addWidget(annotate1, 1, Qt.AlignCenter)
        layout.addWidget(para2, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image2, 1, Qt.AlignCenter)
        layout.addWidget(annotate2, 1, Qt.AlignCenter)
        layout.addWidget(para3, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image3, 1, Qt.AlignCenter)
        layout.addWidget(annotate3, 1, Qt.AlignCenter)
        layout.addWidget(para4, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image4, 1, Qt.AlignCenter)
        layout.addWidget(annotate4, 1, Qt.AlignCenter)
        layout.addWidget(para5, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image5, 1, Qt.AlignCenter)
        layout.addWidget(annotate5, 1, Qt.AlignCenter)
        layout.addWidget(para6, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image6, 1, Qt.AlignCenter)
        layout.addWidget(annotate6, 1, Qt.AlignCenter)
        layout.addWidget(para7, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image7, 1, Qt.AlignCenter)
        layout.addWidget(annotate7, 1, Qt.AlignCenter)
        layout.addWidget(image8, 1, Qt.AlignCenter)
        layout.addWidget(annotate8, 1, Qt.AlignCenter)
        layout.addWidget(para9, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(image9, 1, Qt.AlignCenter)
        layout.addWidget(annotate9, 1, Qt.AlignCenter)
        layout.addWidget(para10, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(title2, 1, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(para11, Qt.AlignLeft | Qt.AlignTop)

        widget = QWidget()
        widget.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: 0px; border-radius: 0px;")

        lay_out = QVBoxLayout()
        lay_out.addWidget(scroll)
        tip.setLayout(lay_out)
        tip.setMinimumSize(1000, 361 * 2)
        tip.exec_()


class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)

        # 设置input表单
        input_form = QFormLayout()
        self.input_box = QLineEdit()
        self._open_file_action = self.input_box.addAction(
            QApplication.style().standardIcon(QStyle.SP_DirOpenIcon), QLineEdit.TrailingPosition
        )  # 添加路径打开按钮
        self._open_file_action.triggered.connect(self.on_open_file)  # 打开并写入路径
        input_form.addRow("    数据样本(*.csv)来自 ：", self.input_box)

        # 设置data_error水平分布
        data_error_h_box = QHBoxLayout()
        self.sb1 = QDoubleSpinBox()
        self.sb1.setDecimals(3)
        self.sb1.setFixedWidth(75)
        self.sb1.setRange(-10, 10)
        self.sb1.setValue(0.000)
        self.sb1.setSingleStep(0.001)
        self.sb1.hide()
        self.cb1 = QComboBox()
        self.cb1.addItems([' 自动学习', ' 无误差', ' 自定义'])  # 设置下拉列表中的选项
        self.cb1.setFixedWidth(100)
        self.cb1.currentIndexChanged.connect(self.error_selected)
        self.sb1.valueChanged.connect(self.get_alpha)
        label1 = QLabel()
        label1.setText("    样本数据误差：")
        label1.setToolTip('优化器会从样本数据中自动学习到误差（不太准，但无需重复实验），可以选择去掉这个误差，也可以自定义误差。\n'
                          '这里误差的现实含义为目标值方差的标准化量（z-score），最好通过重复实验来计算。\n'
                          '\n'
                          '自定义误差为人主观上对实验的判断，取值一般在-自动学习（数值）到10之间，简单参考如下：\n'
                          '  -自动学习：绝对无误差，每次自动学习的误差都可能不同，大多数情况下绝对无误差是不存在的；\n'
                          '  0.0：”正常手感，但可能因仪器或天气存在一些无法避免或我自己都没发现的小小问题,基本不影响实验结论，同“无误差”“；\n'
                          '  0.1：”手感只能说还行吧，或者感觉仪器有点问题，结果应该大差不差“；\n'
                          '  0.5：”这几天手感不好，做实验时总有些不在状态，仪器也总出毛病，感觉和理论上会有挺大偏差的“；\n'
                          '  10：“垃圾仪器，每次测都差了十万八千里，还有几个甚至数量级都不对，要不算了重开吧”。')
        data_error_h_box.addWidget(label1)
        data_error_h_box.addWidget(self.cb1)
        data_error_h_box.addWidget(self.sb1, 1, Qt.AlignLeft)
        data_error_h_box.addStretch(20)

        # 设置set_acq水平分布
        set_acq_h_box = QHBoxLayout()
        self.sb2 = QDoubleSpinBox()
        self.sb2.setDecimals(3)
        self.sb2.setFixedWidth(75)
        self.sb2.setRange(0, 1)
        self.sb2.setValue(0.100)
        self.sb2.setSingleStep(0.001)
        self.cb2 = QComboBox()
        self.cb2.addItems([' q-EI', ' q-UCB'])  # 设置下拉列表中的选项
        self.cb2.setFixedWidth(79)
        self.cb2.currentIndexChanged.connect(self.acq_selected)
        label2 = QLabel()
        label2.setText("    采样方法：")
        label2.setToolTip('采样方法是贝叶斯优化的核心，暗藏着探索（未知）和利用（已知）间的平衡。\n'
                          '此处包含预期提升法（EI）和置信上界法（UCB），q表示单次可同时进行多个实验（或单个实验）。\n'
                          '  q-EI：多点预期提升法，通俗理解，其计算的是下一步优化结果比目前最好的样本还要好的可能性。\n'
                          '        对于完全黑箱实验，这个方法相对能更好地找到最优。\n'
                          '  q-UCB：多点置信上界法，通俗理解，其计算的是一定置信度下，待优化样本的最好表现。\n'
                          '        如果目标值与各因素的相关性比较明显（不需要知道具体相关性），则选这个方法会更好。')
        label3 = QLabel()
        label3.setText("  探索因子：")
        label3.setToolTip('探索因子是采样函数中的重要参数，探索因子越高越倾向于在未知空间中”探索宝藏“。\n'
                          '不同采样函数适配不同大小的探索因子。\n'
                          '  对于q-EI：实际意义类似于以一定概率往非优方向探索。\n'
                          '         有论文表明大小为0.1的探索因子在大多数情况下表现良好。\n'
                          '  对于q-UCB：实际意义对应置信度，数值可参考t分布表（自由度n=∞）或标准正态分布表，\n'
                          '         比如2.576对应99%置信度，1.960对应95%置信度等，最小可为零。')
        set_acq_h_box.addWidget(label2)
        set_acq_h_box.addWidget(self.cb2)
        set_acq_h_box.addWidget(label3)
        set_acq_h_box.addWidget(self.sb2, 1, Qt.AlignLeft)
        set_acq_h_box.addStretch(20)

        # 设置最大最小化水平分布
        max_min_h_box = QHBoxLayout()
        direction = QLabel()
        direction.setText("    寻找目标的：")
        maximize = QRadioButton("最大值")
        maximize.setChecked(True)
        self.direction = "最大值"  # 默认往大的找
        minimize = QRadioButton("最小值")
        maximize.setFixedWidth(72)
        minimize.setFixedWidth(72)
        maximize.toggled.connect(self.max_min_state)
        minimize.toggled.connect(self.max_min_state)
        max_min_h_box.addWidget(direction)
        max_min_h_box.addWidget(maximize, 1, Qt.AlignLeft)
        max_min_h_box.addWidget(minimize, 1, Qt.AlignLeft)
        max_min_h_box.addStretch(20)

        # 设置output表单
        output_form = QFormLayout()
        self.output_box = QLineEdit()
        self.output_box.setText(
            QDir.fromNativeSeparators(
                QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
            )
        )  # 设置输出文件夹默认路径
        self._open_folder_action = self.output_box.addAction(
            QApplication.style().standardIcon(QStyle.SP_DirOpenIcon), QLineEdit.TrailingPosition
        )  # 添加路径打开按钮
        self._open_folder_action.triggered.connect(self.on_open_folder)  # 打开并写入路径
        output_form.addRow("    结果输出在(DirsOnly)：", self.output_box)

        # 设置按钮水平分布
        button_h_box = QHBoxLayout()
        self.start_button = QPushButton()
        self.start_button.setText("查看优化建议")
        self.start_button.hide()
        self.start_button.clicked.connect(lambda: self.show_table(self.bo))
        self.view_button = QPushButton()
        self.view_button.setText("优化可视化")
        self.view_button.hide()
        self.view_button.clicked.connect(lambda: self.show_catalogue(self.bo))
        self.rerun_button = QPushButton()
        self.rerun_button.setText("重新运行")
        self.rerun_button.clicked.connect(self.start_running)
        self.rerun_button.hide()
        button_h_box.addWidget(self.start_button)
        button_h_box.addWidget(self.view_button)
        button_h_box.addWidget(self.rerun_button)

        # 设置按钮
        self.run_button = QPushButton()
        self.run_button.setText("确认并开始运行")
        self.run_button.setStyleSheet("font-size: 20px;")
        self.run_button.clicked.connect(self.start_running)

        # 设置总布局
        general_layout = QVBoxLayout()
        shuru = QLabel()
        shuru.setText("输入设置")
        shuru.setStyleSheet("font-size: 21px;")
        general_layout.addWidget(shuru)
        general_layout.addLayout(data_error_h_box)
        general_layout.addLayout(set_acq_h_box)
        general_layout.addLayout(input_form)
        shuchu = QLabel()
        shuchu.setText("输出设置")
        shuchu.setStyleSheet("font-size: 21px;")
        general_layout.addWidget(shuchu)
        general_layout.addLayout(max_min_h_box)
        general_layout.addLayout(output_form)
        general_layout.addWidget(self.run_button)
        general_layout.addLayout(button_h_box)
        self.setLayout(general_layout)

    # 以上用到的方法
    def on_open_file(self):
        dir_path = QFileDialog.getOpenFileName(
            self,  "打开文件", QDir.homePath(), "逗号分隔数据 (*.csv)"
        )[0]

        if dir_path:
            input_dir = QDir(dir_path)
            self.input_box.setText(QDir.fromNativeSeparators(input_dir.path()))

    def on_open_folder(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "打开文件夹", QDir.homePath(), QFileDialog.ShowDirsOnly
        )

        if dir_path:
            output_dir = QDir(dir_path)
            self.output_box.setText(QDir.fromNativeSeparators(output_dir.path()))

    def error_selected(self, i):
        if i == 0:  # 自动学习
            self.sb1.hide()
            BO.gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5) + WhiteKernel(),
                normalize_y=True,
                n_restarts_optimizer=9,
                random_state=self.ensure_rng()
            )
        if i == 1:  # 无噪声
            self.sb1.hide()
            BO.gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                normalize_y=True,
                n_restarts_optimizer=9,
                random_state=self.ensure_rng()
            )
        if i == 2:  # 自定义
            self.sb1.show()
            BO.gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                normalize_y=True,
                n_restarts_optimizer=9,
                random_state=self.ensure_rng()
            )

    def get_alpha(self, alpha):
        BO.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=9,
            random_state=self.ensure_rng()
        )

    def acq_selected(self, i):
        if i == 0:
            self.sb2.setRange(0, 1)
            self.sb2.setValue(0.100)

        if i == 1:
            self.sb2.setRange(0, 10)
            self.sb2.setValue(2.576)

    def max_min_state(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            self.direction = radio_button.text()

    def start_running(self):
        if self.input_box.text() == '' or self.input_box.text()[-4::] != '.csv':
            # 警告对话框设置
            warning = QDialog()
            warning.setWindowTitle("警告")
            warning.setWindowModality(Qt.ApplicationModal)
            tip = QLabel()
            tip.setText('请在 "数据样本(*.csv)来自于 : " 后选择或填写正确路径')
            btn = QPushButton()
            btn.setText("返回")
            btn.setStyleSheet("padding: 2px")
            btn.clicked.connect(warning.close)
            # 设置警告对话框总布局
            layout = QVBoxLayout()
            layout.addWidget(tip)
            layout.addWidget(btn, 1, Qt.AlignRight)
            warning.setLayout(layout)
            warning.exec_()
        else:
            self.job11 = MyThread(target=self.job1)
            self.job11.start()
            # 等待对话框设置
            self.waiting = QDialog()
            self.waiting.setWindowTitle("正在计算中...")
            self.waiting.setWindowModality(Qt.ApplicationModal)

            # 进度条功能实现
            pb = QProgressBar()
            pb.setRange(0, 0)

            # 取消按钮功能实现
            cancel = QPushButton()
            cancel.setText("取消")
            cancel.clicked.connect(self.stop)

            # 设置对话框总布局
            total_layout = QVBoxLayout()
            total_layout.addStretch(1)
            total_layout.addWidget(pb)
            total_layout.addStretch(1)
            total_layout.addWidget(cancel)
            self.waiting.setLayout(total_layout)
            self.waiting.exec_()

            self.job11.join()
            self.bo = self.job11.get_result()
            if self.bo is None:
                pass

    def job1(self):
        # BO获取输入参数
        BO.acq_method = self.cb2.currentText().strip()
        BO.explore_factor = self.sb2.value()
        BO.data_filename = self.input_box.text()
        BO.direction = self.direction
        BO.output_dirs = self.output_box.text()
        # bo
        bo = BO()
        self.waiting.close()
        self.run_button.hide()
        self.start_button.show()
        self.view_button.show()
        self.rerun_button.show()
        return bo

    def stop(self):
        self.waiting.close()
        # 强行停止进程
        tid = ctypes.c_long(self.job11.ident)
        if not inspect.isclass(SystemExit):
            raise TypeError("Only types can be raised (not instances)")
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
            raise SystemError("PyThreadState_SetAsyncExc failed")
        self.job11.join()

    def show_catalogue(self, bo):

        # 目录对话框设置
        catalogue = QDialog()
        catalogue.setWindowTitle("选择可视化对象")
        # max_min_h_box.addStretch(20)

        # 设置左布局
        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()
        for index in range(len(bo.factor_name)):
            # 根据index获取数据
            arg_sort = np.argsort(bo.seads[:, index])
            x_for_lines1 = bo.seads[:, index][arg_sort]
            for_acq1 = bo.seads[:, -4][arg_sort]
            for_upper1 = bo.seads[:, -2][arg_sort]
            for_lower1 = bo.seads[:, -1][arg_sort]
            for_mean1 = bo.seads[:, -3][arg_sort]

            # 根据index获取数据（单因素）
            x_for_lines2 = bo.big[index][:, 0]
            for_upper2 = bo.big[index][:, 2]
            for_lower2 = bo.big[index][:, 3]
            for_mean2 = bo.big[index][:, 1]
            for_acq2 = bo.big[index][:, 4]

            btn1 = QPushButton()
            btn1.setText(bo.factor_name[index])
            btn1.clicked.connect(
                partial(self.show_figure, x_for_lines1, for_upper1, for_lower1, for_mean1, for_acq1, self.bo, index)
            )
            layout1.addWidget(btn1)

            btn2 = QPushButton()
            btn2.setText(bo.factor_name[index] + '（单）')
            btn2.clicked.connect(
                partial(self.show_figure, x_for_lines2, for_upper2, for_lower2, for_mean2, for_acq2, self.bo, index)
            )
            layout2.addWidget(btn2)

        # 设置目录对话框总布局
        layout = QHBoxLayout()
        layout.addLayout(layout1)
        layout.addLayout(layout2)

        catalogue.setLayout(layout)
        catalogue.setFixedWidth(540)
        catalogue.exec_()

    def show_figure(self, x_for_lines, for_upper, for_lower, for_mean, for_acq, bo, index):
        # 图对话框设置
        view = QDialog()
        view.setWindowTitle("优化建议的可视化")
        # table.setWindowModality(Qt.ApplicationModal)

        # 根据index获取观察点和推荐点
        x_for_observations = bo.factor_value[:, index]
        for_observations = bo.target_value

        x_for_recommends = bo.suggestions[:, index]
        for_recommends = bo.expectations

        # 设置数据model
        upper = QtCharts.QLineSeries()
        lower = QtCharts.QLineSeries()
        mean = QtCharts.QLineSeries()
        acq = QtCharts.QLineSeries()
        five_percent = QtCharts.QLineSeries()
        observations = QtCharts.QScatterSeries()
        recommendations = QtCharts.QScatterSeries()

        for i in range(len(x_for_lines)):
            upper.append(x_for_lines[i], for_upper[i])
            lower.append(x_for_lines[i], for_lower[i])
            mean.append(x_for_lines[i], for_mean[i])
            acq.append(x_for_lines[i], for_acq[i])
            five_percent.append(x_for_lines[i], 5)
        for i in range(len(x_for_observations)):
            observations.append(x_for_observations[i], for_observations[i][0])
        for i in range(len(x_for_recommends)):
            recommendations.append(x_for_recommends[i], for_recommends[i][0])

        area = QtCharts.QAreaSeries(upper, lower)
        area.setName("95% confidence interval")
        area.setColor(QColor(231, 101, 26, 127))
        acq_area = QtCharts.QAreaSeries()
        acq_area.setUpperSeries(acq)
        acq_area.setName("EI acquisition score")
        acq_area.setColor(QColor(102, 193, 140, 127))
        mean.setName("mean prediction")
        pen1 = QPen()
        pen1.setStyle(Qt.SolidLine)
        pen1.setColor(QColor(47, 144, 184))
        pen1.setWidth(4)
        mean.setPen(pen1)
        five_percent.setName("limited value below it")
        pen2 = QPen()
        pen2.setStyle(Qt.SolidLine)
        pen2.setColor(QColor(209, 26, 45))
        pen2.setWidth(1)
        five_percent.setPen(pen2)
        observations.setMarkerShape(observations.MarkerShapeCircle)
        observations.setBorderColor(QColor(133, 109, 134))
        observations.setBrush(QBrush(QColor(133, 109, 134)))
        observations.setMarkerSize(12)
        observations.setName("observations")
        observations.hovered.connect(self.hover_point)
        recommendations.setMarkerShape(recommendations.MarkerShapeRectangle)
        pen2 = QPen()
        pen2.setStyle(Qt.SolidLine)
        pen2.setColor(QColor(252, 195, 7))
        pen2.setWidth(3)
        recommendations.setPen(pen2)
        recommendations.setBrush(QBrush(QColor(255, 0, 0)))
        recommendations.setMarkerSize(14)
        recommendations.setName("recommendations")
        recommendations.hovered.connect(self.hover_point)

        figure = QtCharts.QChart()
        figure.addSeries(area)
        figure.addSeries(mean)
        figure.addSeries(observations)
        figure.addSeries(recommendations)
        figure.setTitle("Gaussian Process Regression & Bayesian Optimization")
        figure.setTitleFont(QFont('Arial'))
        # Setting X-axis
        axis_x = QtCharts.QValueAxis()
        axis_x.setLabelFormat("%.2f")
        axis_x.setTitleText(bo.header_list[index])
        axis_x.setTitleFont(QFont('Arial', 10, QFont.Bold))
        axis_x.setRange(min(x_for_lines), max(x_for_lines))
        figure.addAxis(axis_x, Qt.AlignBottom)
        area.attachAxis(axis_x)
        mean.attachAxis(axis_x)
        observations.attachAxis(axis_x)
        recommendations.attachAxis(axis_x)
        # Setting Y-axis
        axis_y = QtCharts.QValueAxis()
        axis_y.setLabelFormat("%.2f")
        axis_y.setTitleText(bo.target_name)
        axis_y.setTitleFont(QFont('Arial', 10, QFont.Bold))
        axis_y.setRange(min(for_lower), max(for_upper))
        figure.addAxis(axis_y, Qt.AlignLeft)
        area.attachAxis(axis_y)
        mean.attachAxis(axis_y)
        observations.attachAxis(axis_y)
        recommendations.attachAxis(axis_y)
        # Setting Legend
        figure.legend().setVisible(True)
        figure.legend().setAlignment(Qt.AlignRight)

        figure2 = QtCharts.QChart()
        figure2.addSeries(acq_area)
        figure2.addSeries(five_percent)
        figure2.setTitle("One Hundred * Acquisition Function （q-EI）")
        figure2.setTitleFont(QFont('Arial'))
        # Setting X-axis
        axis_x2 = QtCharts.QValueAxis()
        axis_x2.setLabelFormat("%.2f")
        axis_x2.setTitleText(bo.header_list[index])
        axis_x2.setTitleFont(QFont('Arial', 10, QFont.Bold))
        axis_x2.setRange(min(x_for_lines), max(x_for_lines))
        figure2.addAxis(axis_x2, Qt.AlignBottom)
        acq_area.attachAxis(axis_x2)
        five_percent.attachAxis(axis_x2)
        # Setting Y-axis
        axis_y2 = QtCharts.QValueAxis()
        axis_y2.setLabelFormat("%.2f")
        axis_y2.setTitleText("Value（%）")
        axis_y2.setTitleFont(QFont('Arial', 10, QFont.Bold))
        axis_y2.setRange(0, 100)
        figure2.addAxis(axis_y2, Qt.AlignLeft)
        acq_area.attachAxis(axis_y2)
        five_percent.attachAxis(axis_y2)
        # Setting Legend
        figure2.legend().setVisible(True)
        figure2.legend().setAlignment(Qt.AlignRight)

        # 设置图组件
        figure_view = QtCharts.QChartView(figure)
        figure_view.setRenderHint(QPainter.Antialiasing)
        figure_view2 = QtCharts.QChartView(figure2)
        figure_view2.setRenderHint(QPainter.Antialiasing)

        # 设置图对话框总布局
        layout = QVBoxLayout()
        layout.addWidget(figure_view)
        win_width = 900
        win_height = 500
        if bo.acq_method == 'q-EI':
            layout.addWidget(figure_view2)
            win_height = win_height * 2 - 12

        view.setLayout(layout)
        view.resize(win_width, win_height)
        view.exec_()

    def hover_point(self, point, state):
        if state:
            try:
                name = self.sender().name()
            except:
                # QCursor.pos()悬停提示文字显示的位置
                name = ""
            QToolTip.showText(QCursor.pos(), "%s\nx: %.2f\ny: %.2f" %
                              (name[0:-1], point.x(), point.y()), msecShowTime=5000)

    def show_table(self, bo):
        # 表格对话框设置
        table = QDialog()
        table.setWindowTitle("下一步优化建议（仅供参考）")

        # 设置表格组件
        tb = QTableView()
        model = QStandardItemModel(bo.num_str.shape[0], bo.num_str.shape[1])
        model.setHorizontalHeaderLabels(bo.header_list)
        v_header_list = ['推荐实验样本' + str(i + 1) for i in range(bo.num_str.shape[0])]
        model.setVerticalHeaderLabels(v_header_list)
        for i in range(bo.num_str.shape[0]):
            for j in range(bo.num_str.shape[1]):
                item = QStandardItem(str(np.around(bo.num_str[i, j], 3)))
                item.setTextAlignment(Qt.AlignCenter)
                model.setItem(i, j, item)
        tb.setModel(model)
        for lie in range(bo.num_str.shape[1]):
            tb.resizeColumnToContents(lie)

        # 设置输出按钮
        output = QPushButton()
        output.setText("一键输出")
        output.setToolTip("结果将输出在：" + "\n" + "    " + bo.output_dirs + "\n" + "可在主界面修改输出目录")
        output.clicked.connect(lambda: self.output_file(bo))

        # 设置表格对话框总布局
        layout = QVBoxLayout()
        layout.addWidget(tb)
        layout.addWidget(output)
        table.setLayout(layout)
        table.resize(tb.width(), tb.height())
        table.exec_()

    @staticmethod
    def output_file(bo):
        np.savetxt(bo.output_dirs, bo.num_str, delimiter=',', header=','.join(bo.header_list))
        tip = QDialog()
        tip.setWindowTitle("提示")
        label = QLabel()
        label.setText("结果已输出在：" + "\n" + "    " + bo.output_dirs + "\n" + "可在主界面修改输出目录")
        layout = QVBoxLayout()
        layout.addWidget(label)
        tip.setLayout(layout)
        tip.exec_()

    @staticmethod
    def ensure_rng(random_state=None):
        """
        Creates a random number generator based on an optional seed.  This can be
        an integer or another random state for a seeded rng, or None for an
        unseeded rng.
        """
        if random_state is None:
            random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        else:
            assert isinstance(random_state, np.random.RandomState)
        return random_state


class MyThread(threading.Thread):
    def __init__(self, target=None):
        super(MyThread, self).__init__()
        self.func = target

    def run(self):
        self.result = self.func()

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None















if __name__ == '__main__':
    main()
#     """
#         Calculate unit cell parameters a.
#
#         :param
#             density(value) : Default is 1.86 g/cm^-3
#             m(value) : Molecular mass. Default is 1009.15 g/mol
#             h(value) : Layer spacing. Default is 3.7 Angstrom
#         :return:
#             a(value) : Angstrom
#     """

# 字体下拉列表
# ft = QFontComboBox()
# ft.setFontFilters(QFontComboBox.AllFonts)
# general_layout.addWidget(ft)
