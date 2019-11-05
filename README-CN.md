[English README](https://github.com/iqiyi/FASPell)

# FASPell

该仓库（根据GNU通用公共许可证v3.0许可）
包含构建当前最佳（到2019年初）中文拼写检查器所需的所有数据和代码，可以以此复现我们的同名论文中的全部实验：

**FASPell: A Fast, Adaptable, Simple, Powerful Chinese Spell Checker
Based On DAE-Decoder Paradigm**  [LINK](https://www.aclweb.org/anthology/D19-5522.pdf)

此论文发表于 the Proceedings of the 2019 EMNLP 
Workshop W-NUT: The 5th Workshop on Noisy User-generated Text。

使用本代码与数据时，请按如下信息引用我们的论文：

    @inproceedings{hong2019faspell,
        title = "{FASP}ell: A Fast, Adaptable, Simple, Powerful {C}hinese Spell Checker Based On {DAE}-Decoder Paradigm",
        author = "Hong, Yuzhong  and
          Yu, Xianguo  and
          He, Neng  and
          Liu, Nan  and
          Liu, Junhui",
        booktitle = "Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT 2019)",
        month = nov,
        year = "2019",
        address = "Hong Kong, China",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/D19-5522",
        pages = "160--169",
    }

## 概述
中文拼写检查（CSC）的任务通常仅考虑对中文文本中的替换错误进行检测和纠正。
其他类型的错误（例如删除/插入错误）相对较少。

FASPell是中文拼写检查器，可让您轻松完成对任何一种中文文本（简体中文文本；
繁体中文文本； 人类论文； OCR结果等）的拼写检查，且拥有最先进的性能。

<img src="model_fig.png" alt="model_fig.png" width="400">

### 性能
下述表格描述了FASPell在SIGHAN15测试集上的性能。

句子级性能为:

|   | 精确率 | 召回率  | 
| ------------- | ------------- | ------------- | 
| 检错  | 67.6% | 60.0% |
| 纠错  | 66.6% | 59.1% |


字符级性能为:

|   | 精确率 | 召回率  | 
| ------------- | ------------- | ------------- | 
| 检错  | 76.2% | 67.1% |
| 纠错  | 73.5% | 64.8% |

这意味着10个错误检测/纠正中大约7个是正确的，并且可以成功检测/纠正10个错误中的6个。
# 使用方法
以下是能够指导您构建中文拼写检查器的步骤指南。

## 依赖

    python == 3.6
    tensorflow >= 1.7
    matplotlib
    tqdm
    java (仅在使用树编辑距离时需要)
    apted.jar (同上，仅在使用树编辑距离时需要)
## 数据准备

在此步骤中，您将在[此处](#数据)下载所有数据。 数据包括拼写检查数据（用于训练和测试）以及用于计算字符相似度的字符特征。

由于FASPell中使用的大多数数据来自其他提供商，所以请注意下载的数据应转换为我们所需的格式。

在仓库中，我们提供了一些示例数据来占位。下载好全部数据后请用相同的文件名覆盖它们。

完成此步骤后，如果您有兴趣，则可以使用以下脚本来计算字符相似度：

    $ python char_sim.py 午 牛 年 千


请注意，FASPell仅采用字符串编辑距离进行计算
相似。 如果您对使用树编辑距离计算相似度感兴趣
，您需要下载（从
[这里](https://github.com/DatabaseGroup/apted)）并编译一个
树编辑距离可执行文件“ apted.jar”到主目录，然后运行：

    $ python char_sim.py 午 牛 年 千 -t

## 训练

我们强烈建议您在实施此步骤之前阅读我们的论文。

共有三个训练步骤（按顺序）。 点击链接
获得他们的详细信息：
1. 预训练掩码语言模型：[请参阅此处](#预训练)
2. 微调训练掩码语言模型：[请参阅此处](#微调训练)
3. 训练CSD过滤器：[请参见此处](#CSD)

## 运行拼写检查器
检查您的目录结构是否如下：

    FASPell/
      - bert_modified/
          - create_data.py
          - create_tf_record.py
          - modeling.py
          - tokenization.py
      - data/
          - char_meta.txt
      - model/
          - fine-tuned/
              - model.ckpt-10000.data-00000-of-00001
              - model.ckpt-10000.index
              - model.ckpt-10000.meta
          - pre-trained/
              - bert_config.json
              - bert_model.ckpt.data-00000-of-00001
              - bert_model.ckpt.index
              - bert_model.ckpt.meta
              - vocab.txt
      - plots/
          ...
      - char_sim.py
      - faspell.py
      - faspell_configs.json
      - masked_lm.py
      - plot.py
      

现在，您应该可以使用以下命令对中文句子进行拼写检查:
    
    $ python faspell.py 扫吗关注么众号 受奇艺全网首播

您还可以检查文件中的句子（每行一个句子）:

    $ python faspell.py -m f -f /path/to/your/file


如要在测试集上测试拼写检查器，请将`faspell_configs.json`中的`"testing_set"`设置为测试集的路径并运行：

    $ python faspell.py -m e

您可以将`faspell_configs.json`中的`"round"`设置为不同的值，并运行上述命令以找到最佳的回合数。

# 数据
## 中文拼写检查数据
1. 人类生成的数据:
    - SIGHAN-2013 shared task on CSC: 
    [LINK](http://ir.itc.ntnu.edu.tw/lre/sighan7csc_release1.0.zip)
    - SIGHAN-2014 shared task on CSC: 
    [LINK](http://ir.itc.ntnu.edu.tw/lre/clp14csc_release1.1.zip)
    - SIGHAN-2015 shared task on CSC: 
    [LINK](http://ir.itc.ntnu.edu.tw/lre/sighan8csc_release1.0.zip)
2. 机器生成的数据:
    - 我们论文中使用的OCR结果:
       - Tst_ocr: [LINK](https://github.com/iqiyi/FASPell/blob/master/data/ocr_test_1000.txt)
       - Trn_ocr: [LINK](https://github.com/iqiyi/FASPell/blob/master/data/ocr_train_3575.txt)
    
要使用我们的代码，拼写检查数据的格式应按照以下例子:

    错误字数	错误句子	正确句子
    0	你好！我是張愛文。	你好！我是張愛文。
    1	下個星期，我跟我朋唷打算去法國玩兒。	下個星期，我跟我朋友打算去法國玩兒。
    0	我聽說，你找到新工作，我很高興。	我聽說，你找到新工作，我很高興。
    1	對不氣，最近我很忙，所以我不會去妳的。	對不起，最近我很忙，所以我不會去妳的。
    1	真麻煩你了。希望你們好好的跳無。	真麻煩你了。希望你們好好的跳舞。
    3	我以前想要高訴你，可是我忘了。我真戶禿。	我以前想要告訴你，可是我忘了。我真糊塗。

## 中文字符特征
我们使用来自两个开放数据库提供的特征。 使用前请检查其许可证。

|   | 数据库名 | 数据链接  | 使用的文件 |
| ------------- | ------------- | ------------- | ------------- |
| 字形特征<sup>※</sup>  | [漢字データベースプロジェクト（汉字数据库项目）](http://kanji-database.sourceforge.net/) | [LINK](https://github.com/cjkvi/cjkvi-ids) | ids.txt |
| 字音特征  | [Unihan Database](https://unicode.org/charts/unihan.html) | [LINK](https://unicode.org/Public/UNIDATA/Unihan.zip) | Unihan_Readings.txt |


※ 请注意，原始 **ids.txt** 本身不提供笔划级别的IDS（出于压缩目的）。 但是，您可以使用树递归（从具有笔画级IDS的简单字符的IDS开始）来为所有字符自己生成笔画级IDS。

可以与我们的代码一起使用的特征文件（`char_meta.txt`）应该具有格式如下：

    unicode编码	字符	CJKV各语言发音	笔划级别的IDS
    U+4EBA	人	ren2;jan4;IN;JIN,NIN;nhân	⿰丿㇏
    U+571F	土	du4,tu3,cha3,tu2;tou2;TWU,THO;DO,TO;thổ	⿱⿻一丨一
    U+7531	由	you2,yao1;jau4;YU;YUU,YUI,YU;do	⿻⿰丨𠃌⿱⿻一丨一
    U+9A6C	马	ma3;maa5;null;null;null	⿹⿱𠃍㇉一
    U+99AC	馬	ma3;maa5;MA;MA,BA,ME;mã	⿹⿱⿻⿱一⿱一一丨㇉灬

其中：
* CJKV各语言发音的字符串遵循格式：`MC;CC;K;JO;V`；
* 当一个语言中的字符是多音字时，可能的发音用`,`分隔；
* 当一个字符不存在某个语言的发音时，用`null`来做占位符。

# 掩码语言模型
## 预训练
如要复现本文中的实验或立即获得预先训练的模型，可以下载 
[预训练好的模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)。

如要自己预训练，请按照
[GitHub repo for BERT](https://github.com/google-research/bert)中的说明操作。

将预训练模型相关所有文件放在`model/pre-trained/`目录下。

## 微调训练
为了产生我们论文所述的微调样本，请运行以下命令

    $ cd bert_modified
    $ python create_data.py -f /path/to/training/data/file
    $ python create_tf_record.py --input_file correct.txt --wrong_input_file wrong.txt --output_file tf_examples.tfrecord --vocab_file ../model/pre-trained/vocab.txt

然后，您需要做的就是继续按照[GitHub BERT for BERT](https://github.com/google-research/bert)中所述的预训练命令来继续训练预训练模型，唯一不同的是使用预训练模型作为初始的checkpoint。

将经过微调的模型的检查点文件放在`model/fine-tuned/`下。 然后，将`faspell_configs.json`中的`"fine-tuned"`设置为微调模型的路径。

# CSD
训练在CSD可能会花费您很多时间，因为比较麻烦。

## 训练CSD时的总体设置

如论文所述，我们需要为每组候选字符手动找到一条过滤曲线。 在此代码中，我们包括一个小hack：每组候选字符都划分为两个子候选字符组，我们需要找到每个子候选字符组的过滤曲线。 划分标准是排名第一的候选字符是否与原始字符不同（即`top_difference=True/False`）。 这个hack很有帮助，因为我们观察top_difference的值对 把握度-相似度 散点图上的候选字符的分布有很大的影响。您在训练过程中应该也会观察到。

我们建议按照如下顺序来为每组子候选字符组来寻找过滤曲线：

    top_difference=True, sim_type='shape', rank=0
    top_difference=True, sim_type='shape', rank=1
    top_difference=True, sim_type='shape', rank=2
            ...        ,       ...       ,   ...
    top_difference=True, sim_type='sound', rank=0
    top_difference=True, sim_type='sound', rank=1
    top_difference=True, sim_type='sound', rank=2
            ...        ,       ...       ,   ...
    top_difference=False, sim_type='shape', rank=0
    top_difference=False, sim_type='shape', rank=1
    top_difference=False, sim_type='shape', rank=2
            ...        ,       ...       ,   ...
    top_difference=False, sim_type='sound', rank=0
    top_difference=False, sim_type='sound', rank=1
    top_difference=False, sim_type='sound', rank=2


要使sim_type ='shape'，需要在`faspell_configs.json`中设置`"visual": 1`和`"phonological": 0`； 要使sim_type ='sound'，您需要设置`"visual":0`和`"phonologic":1`（保持`"union_of_sims": false`，这让字形相似度的散点图上的曲线和字音相似度的散点图上曲线能独立寻找。）。 要使rank = n，请设置`"rank": n`。

在您准备开始训练之前请注意，如果每组都使用一遍掩码语言模型去生成候选字符的话可能会花费您很多时间。 因此，我们建议在训练过程中的第一组时保存生成的候选字符矩阵，然后对后面的组重复使用它们。 将`"dump_candidates"`设置为保存路径可以保存候选字符。 对于以后的组，将`"read_from_dump"`设置为`true`。

## 为每组候选字符训练过滤器的工作流程
对每组`top_difference=True`的子候选字符组，运行：

    $ python faspell.py -m e -t -d

对每组`top_difference=False`的子候选字符组，运行：

    $ python faspell.py -m e -t

然后，您将在目录`plots/`下看到相应的`.png`图。 请注意，还会有一张放大的高把握度、低相似度角（右下角）候选字符分布`.png`图（因为其中候选字符分布过于密集），以帮助您找到最佳曲线。


对于每个子组，您可以使用散点图找到曲线，然后将其作为函数放入`Curves`类中（如果候选对象位于曲线下方，则返回False），然后在`Filter`类的`__init__()`中调用该函数

## 训练完成后的设置

训练完成后，请记住将`"union_of_sims"`更改为`true`（这将强制对两种不同类型的相似度对结果进行并集，而不考虑`"visual"`和`"phonological"`的值）。 将`"rank"`设置为训练中的最高等级。同时，将`"dump_candidates"`更改为`''`（空字符串），将`"read_from_dump"`更改为`false`。
