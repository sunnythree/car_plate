这是一个在不分割字符的情况下直接识别车牌号码的简单项目。
#### 网络模型:
 ![model](https://github.com/sunnythree/car_plate/blob/master/doc/car_plate_rec.png)
#### 使用方法：
第一步：生成训练和测试数据集  
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
第二步：训练
```
    cd pytorch_model
    python3 train.py 30
```
第三步：测试
```
    python3 test.py
```
将会输出准确率，我训练的car_plate_javer.pt模型能达到98.2的准确率（由于训练和测试数据集都是随机生成的，因此可能不同人测试有差异）。
这个准确率不算高，由于我的笔记本算力有限，没能进一步训练更大、更好的模型，不过我想，这各项目已足以证明不分割直接识别车牌的可行性

#### Pull Request
Pull request is welcome.

#### communicate with
QQ group: 704153141


