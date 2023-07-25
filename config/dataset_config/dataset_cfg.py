import numpy as np
import torchio as tio

def dataset_cfg(dataet_name):

    config = {
        'CREMI':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'MEAN': [0.503902],
                'STD': [0.110739],
                'MEAN_DB2_H': [0.505787],
                'STD_DB2_H': [0.115504],
                'PALETTE': list(np.array([
                    [255, 255, 255],
                    [0, 0, 0],
                ]).flatten())
            },
        'GlaS':
            {
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.787803, 0.512017, 0.784938],
                'STD': [0.428206, 0.507778, 0.426366],
                'MEAN_HAAR_H': [0.528318],
                'STD_HAAR_H': [0.076766],
                'MEAN_HAAR_L': [0.579144],
                'STD_HAAR_L': [0.227451],
                'MEAN_HAAR_HHL': [0.542428],
                'STD_HAAR_HHL': [0.142663],
                'MEAN_HAAR_HLL': [0.569150],
                'STD_HAAR_HLL': [0.220854],
                'MEAN_BIOR1.5_H': [0.525711],
                'STD_BIOR1.5_H': [0.076606],
                'MEAN_BIOR2.4_H': [0.516579],
                'STD_BIOR2.4_H': [0.078798],
                'MEAN_COIF1_H': [0.523858],
                'STD_COIF1_H': [0.081001],
                'MEAN_DB2_H': [0.505234],
                'STD_DB2_H': [0.080919],
                'MEAN_DMEY_H': [0.502698],
                'STD_DMEY_H': [0.078861],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'ISIC-2017':
            {
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.699002, 0.556046, 0.512134],
                'STD': [0.365650, 0.317347, 0.339400],
                'MEAN_DB2_H': [0.489676],
                'STD_DB2_H': [0.081749],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'LiTS':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 3,
                'NORMALIZE': tio.ZNormalization.mean,
                'PATCH_SIZE': (112, 112, 32),
                'FORMAT': '.nii',
                'NUM_SAMPLE_TRAIN': 8,
                'NUM_SAMPLE_VAL': 12
            },
        'Atrial':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'NORMALIZE': tio.ZNormalization.mean,
                'PATCH_SIZE': (96, 96, 80),
                'FORMAT': '.nrrd',
                'NUM_SAMPLE_TRAIN': 4,
                'NUM_SAMPLE_VAL': 8
            },
    }

    return config[dataet_name]
