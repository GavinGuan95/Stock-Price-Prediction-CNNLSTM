run ./train.py to generate default results.
The result is generated under: ./data_loader/configs
Modify ./data_loader/configs/config1.json and ./data_loader/configs/config2.json for different configuration settings
for the input/output/window size. 
Modify the architecture in ./model/model.py, use LSTM only by setting self.conv = False, use dilated
convolution by specifying a value bigger than 1 in the self.dilation.

To change the stock under analysis, modify:
the default variable of function extract_features
fname_in=os.path.join("data_loader/original_data/indices/SPY.csv") in data_preprocessor.py

For the non-neural methods, simply call their corresponding files:
./KNN.py
./RandomForest.py
./Linear_Regression.py
