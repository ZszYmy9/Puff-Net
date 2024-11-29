# Puff-Net

You should download retrained vgg model: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing).

For training Puff-Net, your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── train2014
│   ├── 1.png
│   ├── 2.png
│   ├── ...
├── wikiart
│   ├── 1.png
│   ├── 2.png
│   ├── ...
````
If you want to train your PuffNet on a file folder, you can run the command:
````bash
python train.py --batch_size 8 --content_data <content_data> --style_data <style_data> --train True
````

If you want to test your PuffNet on a file folder, you can run the command:
````bash
python train.py --content_data <content_data> --style_data <style_data> --train False
````

If you want to test your PuffNet on a couple of images, you can run the command:
````bash
python test.py
````
You can select the images and pass their paths to line 91 and 92. 
