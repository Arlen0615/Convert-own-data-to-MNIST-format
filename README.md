# Convert-own-data-to-MNIST-format

MNIST is a famous test data with deep learning on the web. This project is convert own data to MNIST format. It may help you to quickly test new DL model without modify too much code. This is update version of [davidflanagan](https://github.com/davidflanagan/notMNIST-to-MNIST)

#### Changes:
1. Not limit with 28 x 28 image resolution, it depends on first input image's resolution.
2. Not limit with 10 class, it depends on how many sub-folders in your target folder.
3. Not limit with gray images, it depend on on first input image's channels.
4. Cancel output file name, use MNIST original file name, instead of **test**, **train** or **ratio number** to decide output
5. Improve **data number**, if data number is 0 or none, it pack whole data in each class.  

Own data should use following strcture to put file, you must to transfer label to index in your code, our output labels is index of alphabetical order.  

![image](https://github.com/Arlen0615/Convert-own-data-to-MNIST-format/blob/master/readme/own_data_structure.png)

### Instructions:
`python convert_to_mnist_format.py target_folder test_train_or_ratio data_number`

***target_folder:*** must give minimal folder path to convert data  
***test_train_or_ratio:*** must define 'test' or 'train' about this data, if you want seperate total data to test and train automatically, you can input one integer for test ratio, e.q. if you input 2, it mean 2% data will become test data  
***data_number:*** if you input 0 or nothing, it convert total images under each label folder.  

## Samples:  
1. `python convert_to_mnist_format.py notMNIST_small test 0`  
mean: use whole data to make test format from notMNIST_small folder  
2. `python convert_to_mnist_format.py notMNIST_small train`  
mean: use whole data to make train format from notMNIST_small folder  
3. `python convert_to_mnist_format.py notMNIST_small 5`  
mean: use whole data to make MNIST format, and 5% is test format, 95% is train format  
4. `python convert_to_mnist_format.py notMNIST_small 10 300`  
mean: each class use 300 images to make MNIST format, and 10% is test, 90% is train

***Important:***  
After create file, you must excute following code  
`gzip data/*ubyte.`

## Use it on your target DL model:
1. copy t10k-labels-idx1-ubyte, t10k-images-idx3-ubyte, train-labels-idx1-ubyte, train-images-idx3-ubyte to MNIST data path that some DL model follow MNIST  

2. Fixed a little source code  
a. Copy [mnist](https://github.com/Arlen0615/Convert-own-data-to-MNIST-format/tree/master/mnist2) folder to root of DL code.  
b. And find related like following:  
    
```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(‘MNIST Path’, one_hot=True)
```
  
Fixed to

```
import mnist2.mnist as mn
mnist = mn.read_data_sets(‘MNIST Path’, one_hot=True, num_classes=18)
#num_classes is how many class you in your data
```

3. Change channels  
If your own data is RGB, then mnist2 will output [batch_size, heigh x width, channels].  
In normal case, yo still need to know how to change with RGB data, or you can reshape to [batch_size, heigh x width x channels]

Enjoy!
