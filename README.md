# DRFENet
traingry.py is the training code. 

If you want the weight of the color image, choose net = DRFENet(channels=3,num_of_layers=17). 

If you want the weight of the gray image, choose net = DRFENet(channels=1,num_of_layers=opt.num_of_layers).

Lines 224 and 226 of the code will automatically generate the .h5 file you need for your training

The gray dataset of train and val and test  can be downloaded at the following link:

link：[https://pan.baidu.com/s/1BzJKf236Fl1Sv7Q4_cyfyQ ](https://pan.baidu.com/s/1BzJKf236Fl1Sv7Q4_cyfyQ )

Code：o83k 

The RGB dataset of train and val and test  can be downloaded at the following link:

link：[https://pan.baidu.com/s/1eDq_lqksZI85__6jmZX6qA?pwd=3sib](https://pan.baidu.com/s/1eDq_lqksZI85__6jmZX6qA?pwd=3sib)

Code：yo1b 

test(gray).py is used to test the performance of the gray image model

test(RGB).py is used to test the performance of the RGB image model

Choose between them individually

net = DRFENet(channels=1) 

net = DRFENet(channels=3)

