import pandas as pd 
import numpy as np 
import math

class PreProcess():
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def pre_process(self, batches):
        goog_data, goog_labels = self.__subset_data()
        goog_bs, labels_bs = self.__batch_data(goog_data, goog_labels, batches)
        goog_norm = [self.__norm_data(d) for d in goog_bs]
        goog_gram = [self.__gram_matrix(d) for d in goog_norm]

        return {'norm_data': goog_norm[:-1], 'labels': labels_bs, 'gram': goog_gram, 'original_data': goog_data}
    
    # --- Helpers --- #
    def __subset_data(self):
        """inplace moddification of data and feature selection"""
        data = self.data.copy() 
        # drop NAs
        data.dropna(inplace=True)

        # to datetime 
        data['date'] = pd.to_datetime(arg = data['date'])
        data.set_index(keys = ['date'], inplace=True)

        # feature selection 
        data = data[['high', 'low', 'adj_close']].astype(np.float32) # drop last valu (no future value for last index)
        label = data['adj_close']
        label_class = 1/2 * (np.sign(label - label.shift(-1)).dropna().to_numpy() + 1) # transform to [0, 1] space
        
        return data[:-1], label_class.astype(int)
        
    def __norm_data(self, data: np.ndarray):
        """
        normalise the data between [-1, 1] to ensure geometry of dot product is captures soley by u*v = cos(theta)
        then apply unit vector normalisation to ensure norm of data vector equals 1 
        """
        # normalise
        data = data.mean(axis = 1) # average price in time frame 
        mins, maxs = data.min(axis=0), data.max(axis=0)
        return 2*(data - mins) / (maxs - mins) -1 

    def __batch_data(self, data: pd.DataFrame, labels: pd.DataFrame, batches: int):
        """get data snapshots"""
        n, c = data.shape
        if batches > len(data):
            raise IndexError('number of batches must be less than data length to avoid 0 length batches')
        remainder = len(data) % batches
        max_even_batches = data[:n - remainder]
        labels = labels[:n - remainder]
        deltat = int(len(labels)/batches)
        return np.array_split(max_even_batches.to_numpy(), batches), np.array(labels[::deltat])

    def __gram_matrix(self, v: np.ndarray):
        """computes the gramian anguler field (GAF) with elements Gij = cos(phi_i + phi_j) and phi_i = cos(xnorm_i)"""
        phi = np.arccos(v)
        return np.vectorize(lambda a, b: math.cos(a+b))(*np.meshgrid(phi, phi, sparse = True))