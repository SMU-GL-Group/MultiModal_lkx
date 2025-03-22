1. 关于光镜肾小球分割，详见[分割处理/cvat分割流程.md]。
2. 数据集格式：
root/
|-分类类别/
         |-KBXXXXXX(病人编号)/
                             |-模态_疾病_病人编号_原始图像名称.jpg
3. 文件说明：
[分割处理]：基于cvat的分割掩模整理得到分割后的光镜肾小球图像。
[交叉验证数据集划分.py]：划分交叉验证数据集，将每折数据索引存为.npy文件。
[MMdataset.py]：数据载入程序(自定义的dataloader)
[modules]：存放跨模态注意力等模块。
[MyModel]：存放各类模型及对应的模型训练程序(run_XXX.py)。
[tools]：存放各类辅助工具，如结果可视化(visualization.py)
[utils]：杂项。
4. 流程：[分割处理]->[交叉验证数据集划分.py]->[MyModel]下的[run_XXX.py]
