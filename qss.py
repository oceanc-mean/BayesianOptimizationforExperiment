# coding=utf-8
# @Time : 2022/4/6 18:14 
# @Author : 黄学海
# @File : qss.py 
# @Software: PyCharm

qss = """
QMenuBar {
    background-color: #888888;
    spacing: 3px; /* spacing between menu bar items */
}

QMenuBar::item {
    /* sets background of menu item. set this to something non-transparent
        if you want menu color and menu item color to be different */
    background-color: #888888;
    color: #FFFFFF;
    font-size: 18px;
}

QMenuBar::item:selected { /* when user selects item using mouse or keyboard */
    background-color: #654321;
}

QMenuBar::item:pressed {
    background: #ABABAB;
}

QLabel {
    font-family: Times New Roman;
    font-size: 17px;
}

QLineEdit{
    font-family: Times New Roman;
    font-size: 16px;
    padding: 3px;
}

QProgressBar::chunk {
    background-color: #CD96CD;
    width: 10px;
    margin: 0.5px;
}

QRadioButton{
    font-size: 16px;
    padding: 3px;
}

QComboBox{
    font-family: Times New Roman;
    font-size: 16px;
    padding: 3px;
}

QDoubleSpinBox{
    font-family: Times New Roman;
    font-size: 16px;
    padding: 3px;
}

QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
    stop: 0 #2ABf9E, stop: 0.75 #009933, stop: 1 #009933);
    border-style: outset;
    border-width: 0px;
    border-radius: 10px;
    border-color: beige;
    font-family: 仿宋;
    font: bold;
    min-width: 10em;
    padding: 10px;
    font-size: 16px;
}

QPushButton:hover {
    background-color: #2ABf9E;
    border-style: outset;
}

QPushButton:pressed {
    border-width: 1px;
    border-style: outset;
}

/*QTableView 左上角样式*/
QTableView QTableCornerButton::section {
    color: red;
    background-color: rgb(64, 64, 64);
    border: 5px solid #f6f7fa;
    border-radius:0px;
    border-color: rgb(64, 64, 64);
}

QTableView {
    color: white;                                       /*表格内文字颜色*/
    gridline-color: #C07010;                             /*表格内框颜色*/
    background-color: rgb(108, 108, 108);               /*表格内背景色*/
    alternate-background-color: rgb(77, 77, 77);
    selection-color: white;                             /*选中区域的文字颜色*/
    selection-background-color: rgb(64, 64, 64);        /*选中区域的背景色*/
    border: 2px groove gray;
    border-radius: 0px;
    padding: 2px 4px;
}

QHeaderView {
    color: white;
    font: bold 10pt;
    background-color: rgb(108, 108, 108);
    border: 0px solid rgb(144, 144, 144);
    border:0px solid rgb(191,191,191);
    border-left-color: rgba(255, 255, 255, 0);
    border-top-color: rgba(255, 255, 255, 0);
    border-radius:0px;
    min-height:29px;
}

QHeaderView::section{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
    stop:0 rgba(80, 80, 80, 255), stop:1 rgba(30, 30, 30, 255));
    color: rgb(240, 240, 240);
    padding-left: 4px;
    border: 1px solid #C07010;
    min-height: 30px;
}

QScrollBar:vertical{
    width:12px;
    border:1px solid rgba(0,0,0,50);
    padding-top:15px;
    padding-bottom:15px;
}
QScrollBar::handle:vertical{
    width:12px;
    border-radius:5px;
    background:rgba(0,0,0,25%);
    min-height:20;
}
QScrollBar::handle:vertical:hover{
    border-radius:5px;
    background:rgba(0,0,0,50%);
    border:0px rgba(0,0,0,25%);
}
QScrollBar::sub-line:vertical{
    height:15px;
    border-radius:5px;
    border-image:url(:/Res/scroll_up.png);
    subcontrol-position:top;
}
QScrollBar::sub-line:vertical:hover{
    height:15px;
    background:rgba(0,0,0,25%);
    subcontrol-position:top;
}
QScrollBar::add-line:vertical{
    height:15px;
    border-radius:5px;
    border-image:url(:/Res/scroll_down.png);
    subcontrol-position:bottom;
}
QScrollBar::add-line:vertical:hover{
    height:15px;
    background:rgba(0,0,0,25%);
    subcontrol-position:bottom;
}
QScrollBar::add-page:vertical{
    background: #F5F5F5;
    border-radius:5px;
}
QScrollBar::sub-page:vertical{
    background: #F5F5F5;
    border-radius:5px;
}
QScrollBar::up-arrow:vertical{
    border-width:0px;
    max-height:16px;
    min-width:17px;
}
QScrollBar::down-arrow:vertical{
    border-style:outset;
    border-width:0px;
}
"""