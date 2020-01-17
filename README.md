这是一个在不分割字符的情况下直接识别车牌号码的简单项目。
##### 网络模型:
 ![model](https://github.com/sunnythree/car_plate/blob/master/doc/car_plate_rec.png)
##### 使用方法：
第一步：生成训练和测试数据集  
```
    cd generateCarPlate
    python3 genCarPlate.py
```
将会在data目录下生成train、test数据集。  
生成的图片示例：  
![pic1](https://github.com/sunnythree/car_plate/blob/master/doc/沪KTKLTZ.jpg)  
![pic2](https://github.com/sunnythree/car_plate/blob/master/doc/津GMQGF7.jpg)  
![pic3](https://github.com/sunnythree/car_plate/blob/master/doc/渝M885B9.jpg)  
![pic4](https://github.com/sunnythree/car_plate/blob/master/doc/蒙ZUTK8T.jpg)  
第二步：训练
```
    cd pytorch_model
    python3 train.py 30
```
第三步：测试
```
    python3 test.py
```
将会输出准确率

