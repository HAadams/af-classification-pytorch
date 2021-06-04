# Atrial Fibrillation Classification From a Short Single Lead ECG Signal
Reference: https://physionet.org/content/challenge-2017/1.0.0/

This repo contains a very simple Convolutional Neural Network solution to classify ECG Signals into one of four categories.
The network contains 13 conv layers only.

### Data
   - The data is from https://physionet.org/content/challenge-2017/1.0.0/
   - There are around 8500 recordings ranging in length between 9 and 60 seconds. 
   - Each recording was sampled at 300Hz. This means a 9 second recording is an array of 2700 ints.
   - There are four classes in the dataset (AF, Normal, Noisy, Other)

### Preprocessing
  - The data comes in .mat format. Scipy was used (scipy.io.loadmat) to extract numpy arrays.
      - More info: https://colab.research.google.com/drive/10AnlrBXZxqW__qvzJD4Qlu_-oN-00Yo7?usp=sharing
  - I divided the data into thre different datasets. Each dataset contains a fixed number of recording lengths (6, 10, 30)
  - Class imbalance is resolved by dividing long recordings into multiple chunks. For example, a 60 second recording contains ten different 6s recordings.

### Training
   - The model was trained for 25 epochs (5 epochs, 5 folds).
   - Adam optimizer with default values was used.
   - CrossEntropy loss function.

### Testing
   - A train/test split (test size = 0.1) was used to hold out some records for offline testing. 

### Results
```
6s dataset
Accuracy      0.8033648790746583
Precision     [0.81069959 0.86057692 0.92094862 0.62753036]
Recall        [0.75769231 0.84433962 0.87593985 0.72769953]
F1 Score      [0.7833002  0.85238095 0.89788054 0.67391304]
ROC AUC       [0.8455610  0.90254870 0.92337130 0.80151900]
 
10s dataset
Accuracy      0.787719298245614
Precision     [0.82317073 0.83783784 0.93877551 0.56081081]
Recall        [0.77586207 0.78151261 0.85185185 0.72173913]
F1 Score      [0.79881657 0.80869565 0.89320388 0.63117871]
ROC AUC       [0.85131487 0.87080064 0.91489651 0.78944099]
 
30s dataet
Accuracy      0.4521276595744681
Precision     [0.04166667 0.         0.84090909 0.83636364]
Recall        [0.28571429 0.         0.62711864 0.37704918]
F1 Score      [0.07272727 0.         0.7184466  0.51977401]
ROC AUC       [0.51578531 0.5        0.7864275, 0.62034277]
```
