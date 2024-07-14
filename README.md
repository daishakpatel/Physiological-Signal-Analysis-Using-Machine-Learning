# Pain Detection from Wearable Physiological Data

## Project Overview
This project focuses on developing a system that can accurately classify pain from physiological data collected from wearable devices. By analyzing different physiological signals, the system aims to identify patterns associated with pain and no pain states.

## Experimental Design

### Script Functionality
The primary script for this project is named `Project2.py`. While the main script handles the core functionality, the project structure is modular, allowing for better organization and maintainability.

#### Command Line Usage
The script takes command line parameters to specify the data type and the data file to be processed:
```
python Project2.py <data_type> <data_file>
```
- `<data_type>`: Specifies the physiological signal to be analyzed. Options include:
  - `dia` – Diastolic BP
  - `sys` – Systolic BP
  - `eda` – EDA
  - `res` – Respiration
  - `all` – Fusion of all data types
- `<data_file>`: The absolute path to the data file.

### Data and Features
The dataset consists of physiological data from 60 subjects, with measurements for both pain and no pain conditions. Each data entry includes:
- Subject ID
- Data Type
- Class (pain or no pain)
- Data (variable length sequences)

For feature extraction, the following statistics are computed for each data type:
- Mean
- Variance
- Minimum
- Maximum

This results in 16 features in total (4 features for each of the 4 data types). For the fusion of all data types, a combined feature set is created.

### Classifier and Validation
The chosen classifier for this project is a Random Forest, selected for its ability to handle high-dimensional and heterogeneous data. The model is trained and tested using 10-fold cross-validation to ensure robust performance metrics.

The output of the script includes:
- Confusion matrix
- Classification accuracy
- Precision
- Recall

These metrics are averaged over all folds to provide a comprehensive evaluation of the model's performance.

## Insights and Findings

### 1. Choice of Classifier
The Random Forest classifier was chosen due to its effectiveness in handling the complexity and variability of physiological data. Its ensemble approach reduces overfitting and improves generalization, which is crucial for this type of data.

### 2. Data Type with Highest Accuracy
The data type with the highest accuracy was systolic blood pressure (`sys`), with an accuracy of 70%, precision of 71.18%, and recall of 70%. The confusion matrix showed an average of:
- True Positives: 4.2
- False Positives: 1.8
- False Negatives: 1.8
- True Negatives: 4.2

Systolic blood pressure is a significant indicator of physiological stress and is commonly elevated during painful experiences, making it a reliable measure for pain detection.

### 3. Performance of Fusion Features
The fusion of all data types (`all`) achieved the highest overall performance with an accuracy of 74.17%, precision of 75.5%, and recall of 74.16%. This approach leverages the complementary nature of different physiological signals, enhancing the prediction accuracy and reliability.

### 4. Variability in Features
The box plot analysis revealed significant variability across the features, which can be attributed to individual differences, pain-induced responses, and measurement noise. This variability is essential for the classifier to learn and generalize effectively.

### 5. Physiological Signal with Most Variability
EDA (electrodermal activity) exhibited the most visual variability in a random instance of the original physiological signals. While EDA is associated with emotional states and stress, including pain-induced responses, its variability can also be influenced by other factors such as anxiety or cognitive effort.

## Conclusion
This project demonstrates the potential of using physiological data from wearable devices to detect pain. The Random Forest classifier proved effective in handling the complex and variable nature of the data, with systolic blood pressure and fused data providing the most accurate pain detection. The insights gained from this project highlight the importance of feature engineering and the benefits of combining multiple data types for improved classification performance.

For further details and analysis, please refer to the accompanying code and documentation.