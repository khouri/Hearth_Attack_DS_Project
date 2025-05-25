import numpy as np
from sklearn.model_selection import train_test_split


class SplitData():

    def __init__(self, partitions):
        self.partitions = partitions
    pass

    def get_three_sets(self, predictors_df, target_df):
        
        train, tmp_data, train_label, tmp_data_label = \
                                        train_test_split(predictors_df, 
                                                         target_df, 
                                                         train_size = self.partitions[0])

        split = self.partitions[1] / np.sum(self.partitions[1:])

        devset, test, devset_label, test_label = \
                                        train_test_split(tmp_data, 
                                                         tmp_data_label, 
                                                         train_size = split)

        print('Training data size: ' + str(train.shape))
        print('Devset data size: '   + str(devset.shape))
        print('Test data size: '     + str(test.shape))
        print(' ')
        
        return(train, train_label, devset, devset_label, test, test_label)
    pass

    def get_two_sets(self, predictors_df, target_df):
        
        train, test, train_label, test_label = \
                        train_test_split(predictors_df, 
                                            target_df, 
                                            train_size = self.partitions[0])

        print('Training data size: ' + str(train.shape))
        print('Test data size: '     + str(test.shape))
        print(' ')
        
        return(train, train_label, test, test_label)
    pass

pass