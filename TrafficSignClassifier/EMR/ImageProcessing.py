#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('configure', '-f {', '    "conf": { \n        "spark.pyspark.python": "python3", \n        "spark.pyspark.virtualenv.enabled": "true", \n        "spark.pyspark.virtualenv.type":"native", \n        "spark.pyspark.virtualenv.bin.path":"/usr/bin/virtualenv"\n    } \n}\n')


# In[ ]:


sc.install_pypi_package('boto3')
sc.install_pypi_package('Pillow')


# In[ ]:


import pickle
import boto3
from PIL import Image
import io


# In[ ]:


def load_pickle_from_s3(bucket, key):
    '''
    Load a pickle file directly from S3 into a Python dictionary.
    '''
    client = boto3.client('s3')
    response = client.get_object(Bucket=bucket, Key=key)
    body = response['Body'].read()
    data = pickle.loads(body, encoding='latin1')
    return data


def save_images_to_s3(bucket, images, labels, base_path):
    '''
    Saves processed images directly to an S3 bucket.
    '''
    s3 = boto3.client('s3')
    for i, (image, label) in enumerate(zip(images, labels)):
        if image.shape[0] == 1:
            image = image.squeeze(0)
            mode = 'L'
        else:
            image = image.transpose(1, 2, 0)
            mode = 'RGB'

        # Convert numpy array to PIL Image
        img = Image.fromarray(image.astype('uint8'), mode)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Construct the S3 path and upload
        object_key = f"{base_path}/{label}/image_{i}.png"
        s3.upload_fileobj(img_buffer, bucket, object_key)


def process_and_save_images(task):
    bucket, pickle_key, base_path = task
    data = load_pickle_from_s3(bucket, pickle_key)

    # Process and save each subset of images
    save_images_to_s3(bucket, data['x_train'],
                      data['y_train'], f'{base_path}/train')
    save_images_to_s3(bucket, data['x_validation'],
                      data['y_validation'], f'{base_path}/validation')
    save_images_to_s3(bucket, data['x_test'],
                      data['y_test'], f'{base_path}/test')


tasks = [('trafficsigndatabucket', 'data0.pickle', 'data0images')]
rdd = sc.parallelize(tasks)
rdd.foreach(process_and_save_images)

print("Images for data0 saved to S3.")

