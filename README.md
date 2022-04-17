# Cell Neuclei Detection using Semantic Segmentation with U-Net

## 1. Summary
The project is to detect the cell nuclei from the biomedical images where the model is trained with the 2018 Data Science Bowl dataset. The cell nuclei vary in shape and sizes, therefore semantic segmentation would be the best approach for this problem.

## 2. IDE and Framework
IDE - Spyder
Frameworks - TensorFlow, Numpy, Matplotlib, OpenCV & Scikit-learn

## 3. Methodology
### 3.1 Input Pipeline
The dataset consist of train and test folder where each folders contains image folder named 'inputs' as the inputs of the model and 'masks' folder that contains image masks for the labels.
The input images are preprocessed with feature scaling and the labels will be repreented as binary values of 0 and 1. The train data is split with the ratio of 80:20 where 80 is the train sets and 20 is the validation set. The 

### 3.2 Model Pipeline
The architecture used in the project is U-Net which consist of downward stack and upward stack. The downward stack act as the feature extractor and the upward stack produces pixel-wise output. The model stureture is shown in figure below
![model structure](https://user-images.githubusercontent.com/100821053/163698920-cfed80c9-8b78-424d-ad57-94b7c006d43c.png)

The model is trained with the batch size of 16 with 20 epochs resulting training accuracy of 97% and validation accuracy of 96%. The figure below shows the plotted graph between the training and validation accuracy and loss.
![Cell Nuclei Detection Train vs Val Accuracy](https://user-images.githubusercontent.com/100821053/163699047-e80e4c38-9d95-4cf9-8058-559fb7cfac05.png)
![Cell Nuclei Detection Train vs Val Loss](https://user-images.githubusercontent.com/100821053/163699048-fbbaf221-5014-4ea5-912e-734477b79742.png)


## 4. Results

The test data result is shown in figure below
![Test Result](https://user-images.githubusercontent.com/100821053/163698893-d067d85a-1006-4c7d-bf7d-c8e031b954f9.png)

The figure below shows the predictions made by the model comapared to the orignial mask to. The results shows very good accuracy from the model predictions.
![Result 1](https://user-images.githubusercontent.com/100821053/163698989-9e53a312-8a74-418d-b312-5fbf2573339a.png)![Result 2](https://user-images.githubusercontent.com/100821053/163698990-86d40199-f7d0-45b4-93f1-f2189cc1863b.png)![Result 3](https://user-images.githubusercontent.com/100821053/163698994-f02728b0-513a-4c7f-bfb1-ac4c2e90fee5.png)




