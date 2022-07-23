
# **American Sign Language (ASL) Letters Real-Time Detection** 🤟
Customized `YOLOv5` for real-time American Language Sign (ASL) letters detection via `PyTorch`, `OpenCV`, Roboflow and `LabelImg`.

<p align="center">
<img src='./Miscellaneous/hello.gif' width='400'/>
</p>

## 📖 **About**
This project originated from a [video](https://www.youtube.com/watch?v=LvNPKwz4Ghw) that I came across on youtube. A woman standing off to the side was supposedly translating every word in American Language Sign (ASL), but it turned out much of what she was signing was nonsense. The deaf community often finds itself in situations where verbal communication is the norm. Also, in many cases access to qualified interpreter services is not available, which can lead to underemployment, social isolation and public health problems.

Therefore, exploiting `PyTorch`, `OpenCV` and a public dataset on Roboflow I trained a customized version of the `YOLOv5` model for real-time ASL letters detection. This is not yet a model that can be used in real life, however, we are on that path.

I trained several variations of the `YOLOv5` model (changing image size, batch size, number of workers, seed, etc) since the model performed excellently on training, validation **and testing** but not in real-time from my webcam. Only after some time I realized that the problem was in that my webcam was capturing completely different frames from the training/validation/test dataset. To verify that this was surely the case I created my own (baby) dataset for only 4 letters ('H', 'E', 'L', 'O'), providing 5 images per letter. For each image I manually added the bounding box and label using the `LabelImg` library, available [here](https://github.com/heartexlabs/labelImg).

The results were eye-popping, especially given the size of the dataset, but it is again a dataset that cannot be generalized to new contexts.

The theory regarding the YOLO model was covered in an earlier repository, available [here](https://github.com/PaulinoMoskwa/Real-Time-Social-Distancing).

## 📝 **Results on ASL dataset**
The dataset is freely available [here](https://public.roboflow.com/object-detection/american-sign-language-letters/1). It has a lot of images: 1512 for training, 144 for validation and 72 for testing. The problem is the type of images. They are very accurate and very clear, yet they look very similar to each other and do not fit new contexts. 

As previously mentioned, I trained a `Yolov5` model (for almost 4h) just to get a model that is unable to recognize almost any letter in a new context. I realized only after several attempts (I changed the image size from 256 to 512, 448 and even 1024, changed the batch size between 16, 32 etc, and even the number of workers) that the problem was the dataset.

<p align="center">
<img src='./Miscellaneous/bad_hello_gif.gif' width='500'/><br>
<i>Attempt of 'Hello' from webcam with <code>YOLOv5</code> trained on the dataset mentioned above.</i>
</p>

The training notebook is available [here](https://github.com/PaulinoMoskwa/Real-Time-Sign-Language/blob/master/Yolov5%20-%20ASL%20dataset/part%201%20-%20Real-Time%20ASL%20Detection%20-%20Training.ipynb).<br>
The real-time testing notebook is available [here](https://github.com/PaulinoMoskwa/Real-Time-Sign-Language/blob/master/Yolov5%20-%20ASL%20dataset/part%202%20-%20Real-Time%20ASL%20Detection%20-%20Testing.ipynb).

-------------------------------

The results reported by the `YOLOv5` model during training are as follows.

First of all, it is possible to visualize:
1. a histogram to see how many elements per label we have
2. a plot of all the boxes in the training images, colored differently for each label, so as to understand whether the sizes of the boxes are sufficiently different (it is convenient to have a variety)
3. a plot of the <span> $(x,y)$ </span> values related to the position of the box within each image (again, it would be good to see fairly scattered points)
4. a plot of the <span> $(width, height)$ </span> values related to the size of the boxes in each image (again, it would be convenient to have fairly scattered points)

<p align="center">
<img src='./Miscellaneous/ASL/labels.jpg' width='700'/><br>
<i>In order: 1. in the upper-left corner, 2. in the upper-right corner, 3. in the lower-left corner and 4. in the lower-right corner.</i>
</p>

More informations about <span> $x, y, \hspace{2pt}width, \hspace{2pt}height$ </span> are also available in an other format.

<p align="center">
<img src='./Miscellaneous/ASL/labels_correlogram.jpg' width='700'/><br>
<i>Labels correlogram.</i>
</p>

It is possible to evaluate how well the training procedure performed by visualizing the logs in runs folder.

<p align="center">
<img src='./Miscellaneous/ASL/results.png' width='900'/><br>
<i>Training history.</i>
</p>

It is also possible to see how good the predictions are and which classes caused the most difficulties.

<p align="center">
<img src='./Miscellaneous/ASL/confusion_matrix.png' width='900'/><br>
<i>Confusion matrix.</i>
</p>

Moreover, it is possible to visualize the precision-recall curve.

<p align="center">
<img src='./Miscellaneous/ASL/PR_curve.png' width='700'/><br>
<i>Precision vs. recall curve.</i>
</p>

Finally, we take a look on some other peculiarities.<br>
The file `train_batch0.jpg` shows train batch 0 mosaics and labels.

<p align="center">
<img src='./Miscellaneous/ASL/train_batch0.jpg' width='700'/>
</p>

Instead, `val_batch0_labels.jpg` shows validation batch 0 labels.

<p align="center">
<img src='./Miscellaneous/ASL/val_batch1_labels.jpg' width='700'/>
</p>

Lastly, `val_batch0_pred.jpg` shows validation batch 0 predictions.

<p align="center">
<img src='./Miscellaneous/ASL/val_batch1_pred.jpg' width='700'/>
</p>


## 📝 **Results on Hello dataset**
To test the validity of my thesis, that is, that the dataset on which I trained `YOLOv5` is not generic enough and does not allow generalization of the model, I created my own dataset. I chose the letters 'H', 'E', 'L', 'O' and for each I took (only) 5 webcam images. After that I created the information regarding the boxes and labels with the `LabelImg` tool. With a total of 20 images (among other things, repeated for both training and validation) I trained a model of `YOLOv5` on 500 epochs.

In conclusion, the model came out outstanding (considering the amount of images used). However, again it is a model that is only great in this context, even less flexible than the previous one. But at least now it is clear that the problem was the dataset and not some component of the training.

<p align="center">
<img src='./Miscellaneous/hello_gif.gif' width='500'/><br>
<i>Attempt of 'Hello' from webcam with <code>YOLOv5</code> trained on my dataset.</i>
</p>

The training notebook is available [here](https://github.com/PaulinoMoskwa/Real-Time-Sign-Language/blob/master/Yolov5%20-%20Personalized%20dataset/part%201%20-%20Real-Time%20Hello%20Detection%20-%20Training.ipynb).<br>
The real-time testing notebook is available [here](https://github.com/PaulinoMoskwa/Real-Time-Sign-Language/blob/master/Yolov5%20-%20Personalized%20dataset/part%202%20-%20Real-Time%20Hello%20Detection%20-%20Testing.ipynb).

-------------------------------

The results reported by this second version of the `YOLOv5` model during training are as follows.

<p align="center">
<img src='./Miscellaneous/Hello/labels.jpg' width='700'/><br>
<i>Informations about the labels - part 1.</i>
</p>

<p align="center">
<img src='./Miscellaneous/Hello/labels_correlogram.jpg' width='700'/><br>
<i>Informations about the labels - part 2 (correlogram).</i>
</p>

Training performance visualization.

<p align="center">
<img src='./Miscellaneous/Hello/results.png' width='900'/><br>
<i>Training history.</i>
</p>

How good the predictions are and which classes caused the most difficulties.

<p align="center">
<img src='./Miscellaneous/Hello/confusion_matrix.png' width='700'/><br>
<i>Confusion matrix.</i>
</p>

<p align="center">
<img src='./Miscellaneous/Hello/PR_curve.png' width='700'/><br>
<i>Precision vs. recall curve.</i>
</p>

Finally, we take a look on some other peculiarities.<br>
The file `train_batch0.jpg` shows train batch 0 mosaics and labels.

<p align="center">
<img src='./Miscellaneous/Hello/train_batch2.jpg' width='700'/>
</p>

Instead, `val_batch0_labels.jpg` shows validation batch 0 labels.

<p align="center">
<img src='./Miscellaneous/Hello/val_batch0_labels.jpg' width='700'/>
</p>

Lastly, `val_batch0_pred.jpg` shows validation batch 0 predictions.

<p align="center">
<img src='./Miscellaneous/Hello/val_batch0_pred.jpg' width='700'/>
</p>

## ✍️ **About `LabelImg`**
`LabelImg` is a (free and easily accessible, thank you 🥰) package for label images for object detection. How does it work? There are a couple of steps to follow.

1. First of all, it is necessary to clone the repository. It is even possible to run the command from a notebook:<br>
```
	!git clone https://github.com/heartexlabs/labelImg
```

2. Next, it is necessary to install two dependencies:<br>
```
	pip install PyQt5
	pip install lxml
```

3. Once done, set up some settings. Always from notebook:<br>
```
	!cd ./labelImg && pyrcc5 -o libs/resources.py resources.qrc
```
4. Now we need to go into the `LabelImg` folder and **move manually** inside the `lib` folder the following files:<br>
```
	resources.py
	resources.qrc
```
5. Go to the command line, activate the correct enviroment (in my case I created a ML enviroment: `C:\Enviroments\ML\Scriptsactivate.bat`) and go into the `LabelImg` folder:<br>
```
    cd ..\labelImg
```

6. Run `LabelImg`:<br>
```
    python labelImg.py
```
7. Select `Open Dir` and open the directory where all the images are

8. Select `Change Save Dir` and open the directory where all label information will be saved

9. Check that the selected format is correct, in this case I used `YOLO` (depending on the model you use, the format of notations is different)

10. Select `View` and then `Autosave Mode mode`, so as to automatically save the labels

11. Use the letter `W` to create the label and move between images using `A` and `D`

12. OPTIONAL: inside `LabelImg` there were also other labels. It does not harm the procedure, however they can be removed if desired. In the output folder (`Change Save Dir`) there is a `.txt` with the classes. It is sufficient to edit this file by deleting unnecessary classes. Beware, however, that we then have to re-edit all the `.txt` files because obviously the association of the classes changes.<br>
For example, if before we had `['dog', 'cat', 'A', 'B']` and now we have `['A', 'B']`, we will have to edit the `.txt` by changing all the files corresponding to `A` by removing the value `2` (old position of `A` in the class list) and putting the value `0` (new position of `A` in the class list), and so on

13. Finally, it is necessary to create a `data.yaml` file, which will be used by the model to figure out where to find all the data. The file in this case (already adapted to colab training) is:<br>
```
    train: /content/drive/MyDrive/ASL/PAULO/images
    val: /content/drive/MyDrive/ASL/PAULO/images

    nc: 4
    names: ['E', 'H', 'L', 'O']
```