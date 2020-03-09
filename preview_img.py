from prepare_data import *
from train import process_features
from configuration import *
import cv2


train_dataset, valid_dataset, test_dataset_batch, train_count, valid_count, test_count = generate_datasets()
for features in train_dataset:
    images, labels = process_features(features)
    dataset = tf.data.Dataset.from_tensor_slices((images.numpy(), labels.numpy()))
    for image, label in dataset:
        print(image, label)

    # for i in range(BATCH_SIZE):
    #     image = images[i]
    #     label = labels[i]
    #     cv2.imshow(str(label), image.numpy())
    #     cv2.waitKey()
    break

print('done')
