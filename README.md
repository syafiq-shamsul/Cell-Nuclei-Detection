# Cell Neuclei Detection using Semantic Segmentation with U-Net

## 1. Summary
The project is to detect the cell nuclei from the biomedical images where the model is trained with the 2018 Data Science Bowl dataset. The cell nuclei vary in shape and sizes, therefore semantic segmentation would be the best approach for this problem.

## 2. IDE and Framework
IDE - Spyder
Frameworks - TensorFlow, Numpy, Matplotlib, OpenCV & Scikit-learn

## 3. Methodology
### 3.1 Input Pipeline
The dataset consist of train and test folder where each folders contains image folder named 'inputs' as the inputs of the model and 'masks' folder that contains image masks for the labels.
The input images are preprocessed with feature scaling and the labels will be repreented as binary values of 0 and 1. The train data is split with the ratio of 80:20 where 80 is the train sets and 20 is the validation set.

### 3.2 Model Pipeline
The architecture used in the project is U-Net which consist of downward stack and upward stack. The downward stack act as the feature extractor and the upward stack produces pixel-wise output. The model is trained with the batch size of 16 with 20 epochs resulting training accuracy of 97% and validation accuracy of 96%

## 4. Results

The test data result is shown in figure below
![Test Result](https://user-images.githubusercontent.com/100821053/163698893-d067d85a-1006-4c7d-bf7d-c8e031b954f9.png)

