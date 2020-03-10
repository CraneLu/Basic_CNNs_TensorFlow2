from prepare_data import *
from train import process_features
from configuration import *
import cv2

# j = 0
# k = 0
# train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
# # 3039 392 374
# print(valid_dataset)

# for features in valid_dataset:
#     print(features)
#     j += 1
#     print('batches', j)
#     images, labels = process_features(features)
#     for i in range(BATCH_SIZE):
#         k += 1
#         print('total', k)
#         image = images[i]
#         label = labels[i]
#         if image is None or label is None or label=='':
#             print(image, label)
#
# print('done')

test_dataset = get_parsed_dataset(tfrecord_name='dataset/testdata.tfrecord')
for data in test_dataset.take(2):
    print(data)

