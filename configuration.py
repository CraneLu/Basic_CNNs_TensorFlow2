# some training parameters
EPOCHS = 10
BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 1024
NUM_CLASSES = 12
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3
SAVE_MODEL_DIR = "saved_model/"
SAVE_EVERY_N_EPOCH = 5
DATASET_DIR = "dataset/"
TRAIN_DIR = DATASET_DIR + "train"
VALID_DIR = DATASET_DIR + "valid"
TEST_DIR = DATASET_DIR + "test"
TRAIN_TFRECORD = DATASET_DIR + "train.tfrecord"
VALID_TFRECORD = DATASET_DIR + "valid.tfrecord"
TEST_TFRECORD = DATASET_DIR + "test.tfrecord"
# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.8
TEST_SET_RATIO = 0.1

# choose a network
# 0: mobilenet_v1, 1: mobilenet_v2, 2: mobilenet_v3_large, 3: mobilenet_v3_small
# 4: efficient_net_b0, 5: efficient_net_b1, 6: efficient_net_b2, 7: efficient_net_b3
# 8: efficient_net_b4, 9: efficient_net_b5, 10: efficient_net_b6, 11: efficient_net_b7
# 12: ResNeXt50, 13: ResNeXt101
# 14: InceptionV4, 15: InceptionResNetV1, 16: InceptionResNetV2
# 17: SE_ResNet_50, 18: SE_ResNet_101, 19: SE_ResNet_152
# 20: SqueezeNet
# 21: DenseNet_121, 22: DenseNet_169, 23: DenseNet_201, 24: DenseNet_269
# 25 ~ 28 : ShuffleNetV2 (0.5x, 1.0x, 1.5x, 2.0x)
# 29: ResNet_18, 30: ResNet_34, 31: ResNet_50, 32: ResNet_101, 33: ResNet_152

# EfficientNets:
# b0 = (1.0, 1.0, 224, 0.2)
# b1 = (1.0, 1.1, 240, 0.2)
# b2 = (1.1, 1.2, 260, 0.3)
# b3 = (1.2, 1.4, 300, 0.3)
# b4 = (1.4, 1.8, 380, 0.4)
# b5 = (1.6, 2.2, 456, 0.4)
# b6 = (1.8, 2.6, 528, 0.5)
# b7 = (2.0, 3.1, 600, 0.5)
model_index = 3

