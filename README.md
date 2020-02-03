这是一个在不分割字符的情况下直接识别车牌号码的简单项目。
#### 使用方法：
##### 第一步：生成训练和测试数据集  
```
    cd generateCarPlate
    python3 genCarPlate.py
```
将会在data目录下生成train、test数据集。  
生成的图片示例：  
![沪KTKLTZ](https://github.com/sunnythree/car_plate/blob/master/doc/沪KTKLTZ.jpg)  
![津GMQGF7](https://github.com/sunnythree/car_plate/blob/master/doc/津GMQGF7.jpg)  
![渝M885B9](https://github.com/sunnythree/car_plate/blob/master/doc/渝M885B9.jpg)  
![蒙ZUTK8T](https://github.com/sunnythree/car_plate/blob/master/doc/蒙ZUTK8T.jpg)  
生成的图片已经做了旋转、模糊、灰度变化等图像增强。git

### 训练卷积only的模型
##### 网络模型:
 ![model](https://github.com/sunnythree/car_plate/blob/master/doc/car_plate_rec.png)
##### 训练
```
    cd pytorch_model
    python3 train.py 30 0.0001
```
30是在训练集训练的次数，0.0001是学习速率  
##### 测试
```
    python3 test.py
```
将会输出准确率，我训练的car_plate_javer.pt模型能达到98.2的准确率（由于训练和测试数据集都是随机生成的，因此可能不同人测试有差异）。
这个准确率不算高，由于我的笔记本算力有限，没能进一步训练更大、更好的模型，不过我想，这个项目已足以证明不分割直接识别车牌的可行性

### 训练crnn(双向gru)+ctc
##### 网络模型:
 ![model](https://github.com/sunnythree/car_plate/blob/master/doc/crnn-ctc.png)
##### 训练
```
    cd pytorch_model
    python3 train.py 30 0.0001 10
```
30是在训练集训练的次数，0.0001是学习速率,10是batch的大小
##### 测试
```
    python3 test.py
```
#### Pull Request
Pull request is welcome.

#### communicate with
QQ group: 704153141

#### 致谢
数据集生成使用了[genCarPlate](https://github.com/derek285/generateCarPlate)，感谢作者！


#### License
BSD 3-Clause

