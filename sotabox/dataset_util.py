print("[INFO] dataset_util")

def normalize(train_data, test_data, use_channel_mean=True):

    if use_channel_mean:
        mean = np.mean(train_data, axis=[0,1,2])
        std  = np.std(train_data, axis=[0,1,2])
        train_data = (train_data - mean)/std
        test_data = (test_data - mean)/std
    else:
        train_data /= 255.
        test_data /= 255.

    return train_data, test_data
