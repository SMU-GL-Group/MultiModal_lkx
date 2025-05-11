import numpy as np
import matplotlib.pyplot as plt


class Radar(object):

    def __init__(self, figure, labels, scale, lim, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(labels)
        self.angles = np.arange(0, 360, 360.0 / self.n)
        self.label_angles = np.arange(0, 355, 76)
        self.lim = lim

        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=[], fontsize=20)  # 设置极角轴标签 # 空label,后期再画上

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)

        for ax, angle, label, i in zip(self.axes, self.label_angles, scale, lim):
            ax.set_rgrids(i[1:], angle=angle, labels=[], fontsize=20)  # 设置极坐标轴的径向网格线和标签。 圆心不显示刻度所以i[1:]
            ax.spines['polar'].set_visible(False)
            ax.set_ylim(i[0], i[-1])  # 这里应该是对应轴的刻度
            ax.grid(False)  # 隐藏圆形网格线
        # 微调每个坐标轴
        # self.axes[0].set_rgrids(lim[0][1:], angle=0,   labels=scale[0], fontsize=22)
        # self.axes[1].set_rgrids(lim[1][1:], angle=73,  labels=scale[1], fontsize=22)
        # self.axes[2].set_rgrids(lim[2][1:], angle=144, labels=scale[2], fontsize=22)
        # self.axes[3].set_rgrids([93.4,94.4,95.4,96.4], angle=213, labels=scale[3], fontsize=22)
        # self.axes[4].set_rgrids(lim[4][1:], angle=289, labels=scale[4], fontsize=22)
        # ax.yaxis.grid(True, linewidth=1.5, color='gray', linestyle='-', alpha=0.7)  # 设置每环线的属性

    def plot(self, values, *args, **kw):  # *args位置参数, **kw关键字参数
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])  # 度数转为弧度
        limits = []
        # 把输入的值映射到坐标轴的区间内(以第一个极坐标轴为量纲, 其余坐标轴减去各自坐标轴第一环刻度,除以各自单位刻度,乘以量纲单位刻度0.5,再加上量纲的坐标轴的第一环)
        for idx in range(1, len(values)): # 从第二个坐标轴开始
            limits.append((values[idx] - self.lim[idx][1])/(self.lim[idx][1]-self.lim[idx][0]))
        for idx in range(len(values)-1):  # 除第一个极坐标外，其他极坐标都需要进行映射
            values[idx+1] = (limits[idx]) * 1 + self.lim[0][1]  # todo：需要修改量纲刻度
        values = np.r_[values, values[0]]  # 闭合多边形
        self.ax.plot(angle, values, *args, **kw)
        for j in np.arange(92, 96.5, step=1):
            self.ax.plot(angle, 6 * [j], '-', lw=0.8, color='black')
        for j in range(5):
            self.ax.plot([angle[j], angle[j]], [92, 100], linestyle='-', lw=0.8, color='black')
        # 自定义填充颜色和透明度
        fill_color = kw.get('color')
        self.ax.fill(angle, values, color=fill_color, alpha=0.4)


if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 10))

    # 准备数据
    # α、β、γ的组合
    label = ['ACC','AUC','Precision','Recall','F1-score'] # 极坐标轴的标签

    y_scale = [  # 每个极坐标轴(除圆心外)的刻度标签/文本
        list(('93',   '94', '95',   '96')),
        list(('98.375', '98.75', '99.125', '99.5')),
        list(('93',   '94', '95',   '96')),
        list(('93',   '94', '95',   '96')),
        list((' 93',  '94', '95',   '96')),
    ]

    y_lim = np.array([ # 每个极坐标轴的取值范围。精度需一致，否则计算可能有误
        [92.00, 93.00, 94.00, 95.00, 96.00],
        [98.00, 98.375, 98.75, 99.125, 99.50],
        [92.00, 93.00, 94.00, 95.00, 96.00],
        [92.00, 93.00, 94.00, 95.00, 96.00],
        [92.00, 93.00, 94.00, 95.00, 96.00]
    ])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    radar = Radar(fig, labels=label, scale=y_scale, lim=y_lim)

    # 分别绘制每组数据
    # 每列对应Accuracy、AUC、Precision、Recall、F1-score的均值
    radar.plot([94.44, 98.83, 94.37, 94.69, 94.41], '-', lw=2, alpha=1.0, label='α=0.5 β=0.3 γ=0.2', color='orange'   )
    radar.plot([93.70, 98.35, 93.53, 93.80, 93.63], '-', lw=2, alpha=1.0, label='α=0.5 β=0.2 γ=0.3', color='#7F3346'  )
    radar.plot([92.78, 98.86, 92.73, 93.13, 92.71], '-', lw=2, alpha=1.0, label='α=0.2 β=0.3 γ=0.5', color='blue' )
    radar.plot([94.07, 98.75, 94.13, 94.40, 94.11], '-', lw=2, alpha=1.0, label='α=0.2 β=0.5 γ=0.3', color='#33A2FF')
    radar.plot([93.89, 98.59, 93.87, 94.00, 93.79], '-', lw=2, alpha=1.0, label='α=0.3 β=0.2 γ=0.5', color='#B34FFF')
    radar.plot([95.37, 99.05, 95.29, 95.52, 95.32], '-', lw=2, alpha=1.0, label='α=0.3 β=0.5 γ=0.2', color='#4FFF33'  )

    #radar.ax.legend(loc='best', fontsize=15)
    plt.show(dpi=300)
    # plt.savefig('fig-3.png', dpi=300)
