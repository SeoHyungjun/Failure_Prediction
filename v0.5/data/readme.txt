All samples were collected from an enterprise-class disk model of Seagate named ST31000524NS. There are samples from 23,395 disks in our dataset. Each disk was labeled good or failed, with only 433 disks in the failed class and the rest of disks (22,962) in the good class. This dataset was very unbalanced as the number of good disks was more than 50 times that of failed disks. SMART Attribute values were read per hour for each disk. For good disks, the samples in a week's period are kept in the dataset, so every good disk has 168 samples. For failed disks, samples in a longer time period (20 days before actual failure) are saved. Samples may be less than 480 for failed disks if they didn't survive 20 days of operation since we began to collect data.
Every attribute value has been scaled to the same interval [-1, 1] and their exact values withhold. The serial-number of the disk is replaced by a number ranging from 1 to 23,395.

Each line in the dataset contains 14 columns which are seperated by commas. The meaning of each column is listed as follows.
Column 1 : index of the disk representing it's serial-number, ranging from 1 to 23,395.
Column 2 : class label of the disk, it is -1 for disks that are failed and +1 for disks that are good.
Column 3 : VALUE of SMART ID #1 , Raw Read Error Rate
Column 4 : VALUE of SMART ID #3 , Spin Up Time
Column 5 : VALUE of SMART ID #5 , Reallocated Sectors Count
Column 6 : VALUE of SMART ID #7 , Seek Error Rate
Column 7 : VALUE of SMART ID #9 , Power On Hours
Column 8 : VALUE of SMART ID #187 , Reported Uncorrectable Errors
Column 9 : VALUE of SMART ID #189 , High Fly Writes
Column 10 : VALUE of SMART ID #194 , Temperature Celsius
Column 11 : VALUE of SMART ID #195 , Hardware ECC Recovered
Column 12 : VALUE of SMART ID #197 , Current Pending Sector Count
Column 13 : RAW_VALUE of SMART ID #5 , Reallocated Sectors Count
Column 14 : RAW_VALUE of SMART ID #197 , Current Pending Sector Count

As mentioned above, column 3 to 13 have been normailized to the same interval [-1, 1] and their exact values withhold. Due to intellectual property, we can not give the model training and failure detection program that we used in this paper. However, there are many open source implementations of SVM and BP neural netwrok algorithms. We the SVM implementation named LIBSVM in our paper, LIBSVM is available at http://www.csie.ntu.edu.tw/~cjlin/libsvm/. Besides, an implementation of BP algorithm in C++ can be seen in http://www.codeproject.com/Articles/13582/Back-propagation-Neural-Net.

Contributed by:  Bingpeng Zhu  (nkuzbp@hotmail.com)
                 Gang Wang     (wgzwp@163.com)
                 Xiaoguang Liu (Liuxg74@yahoo.com.cn)
                 Dianming Hu   (hudiangming@baidu.com)
                 Sheng Lin     (shshsh.0510@gmail.com)
                 Jingwei Ma    (majingweitom@yahoo.com)

                 Nankai University & Baidu Inc.
                 January 8, 2013

The website of our Lab (Nankai-Baidu Joint Lab) is: http://nbjl.nankai.edu.cn/.