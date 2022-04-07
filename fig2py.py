# coding=utf-8
# @Time : 2022/4/6 22:17 
# @Author : 黄学海
# @File : fig2py.py 
# @Software: PyCharm

import base64

with open("figure.py", "w") as f:
    f.write('class Figure(object):\n')
    f.write('\tdef __init__(self):\n')
    f.write("\t\tself.img='")

with open("./image/icon/last.ico", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img1='")

with open("./image/emoji/SomeoneLikeMe.jpeg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img2='")

with open("./image/figure/GaussianProcess.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img3='")

with open("./image/figure/EI.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img4='")

with open("./image/figure/UCB.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img5='")

with open("./image/figure/starUI.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img6='")

with open("./image/figure/tooltip.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img7='")

with open("./image/figure/calculating.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img8='")

with open("./image/figure/calculated.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img9='")

with open("./image/figure/output.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img10='")

with open("./image/figure/chose.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img11='")

with open("./image/figure/multi_factor.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)


with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img12='")

with open("./image/figure/single_factor.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)

with open("figure.py", "a") as f:
    f.write("'\n")
    f.write("\t\tself.img13='")

with open("./image/figure/format.jpg", "rb") as i:
    b64str = base64.b64encode(i.read())
    with open("figure.py", "ab+") as f:
        f.write(b64str)

with open("figure.py", "a") as f:
    f.write("'")
