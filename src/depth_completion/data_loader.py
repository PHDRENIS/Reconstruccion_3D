import os
import numpy as np
import cv2
import tensorflow as tf

class SunRGBDGenerator(tf.keras.utils.Sequence):
    def __init__(self, rgb_dir, depth_input_dir, mask_dir, depth_gt_dir, batch_size=8, img_size=(480, 640), shuffle=True):
        self.rgb_dir = rgb_dir
        self.depth_input_dir = depth_input_dir
        self.mask_dir = mask_dir            
        self.depth_gt_dir = depth_gt_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        
        self.filenames = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.indexes = np.arange(len(self.filenames))
        
    def __len__(self):
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_filenames_temp = [self.filenames[k] for k in indexes]
        X, y = self.__data_generation(list_filenames_temp)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_filenames_temp):
        # X shape: (Batch, 480, 640, 5) -> RGB(3) + Depth(1) + Mask(1)
        X = np.empty((self.batch_size, *self.img_size, 5), dtype=np.float32)
        y = np.empty((self.batch_size, *self.img_size, 1), dtype=np.float32)

        for i, filename in enumerate(list_filenames_temp):
            name_no_ext = os.path.splitext(filename)[0]

            # RGB
            rgb = cv2.imread(os.path.join(self.rgb_dir, filename))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype('float32') / 255.0

            # DEPTH INPUT (npy)
            depth_in = np.load(os.path.join(self.depth_input_dir, name_no_ext + ".npy"))
            depth_in = np.expand_dims(depth_in, axis=-1)

            # MASK (PNG)
            mask_path = os.path.join(self.mask_dir, name_no_ext + ".png")
            
            # We verify the mask exists
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"[CRITICAL ERROR] Missing mask: {mask_path}\n"
                                        f"The generator tried to load it because the RGB exists: {filename}")

            mask = cv2.imread(mask_path, 0) 
            
            if mask is None:
                raise ValueError(f"[READ ERROR] The file exists but OpenCV could not read it (is it corrupted?): {mask_path}")

            # Normalize: 0 stays 0, 255becomes 1.0
            mask = (mask > 0).astype('float32') 
            mask = np.expand_dims(mask, axis=-1)

            # COmbine into input tensor of 5 channels
            X[i,] = np.concatenate([rgb, depth_in, mask], axis=-1)

            # GT Depth (ground truth)
            depth_gt = np.load(os.path.join(self.depth_gt_dir, name_no_ext + ".npy"))
            y[i,] = np.expand_dims(depth_gt, axis=-1)

        return X, y