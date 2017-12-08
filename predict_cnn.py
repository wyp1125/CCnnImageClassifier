import tensorflow as tf
import numpy as np
import json
import os,sys,glob,cv2,re
import pprint

if len(sys.argv)<4:
    print("python3 predict_cnn.py json_configuration_file image_folder output")
    quit()

#Load the configuration file to get basic paramters
mdata=json.load(open(sys.argv[1]))
classes=sorted(list(mdata["classes"].keys()))
num_classes=len(classes)
print(classes)
files=os.listdir(sys.argv[2])
image_size=128
num_channels=3

#Read new images
images = []
for eachfile in files:
    if re.search(".jpg",eachfile,re.IGNORECASE):
        image = cv2.imread(sys.argv[2]+"/"+eachfile)
        image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
        images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)
num_image=images.shape[0]
y_test_images = np.zeros((num_image, num_classes))

#Load the established neural network model
sess = tf.Session()
saver = tf.train.import_meta_graph('run1/cat_doc_cow_classifier.meta')
saver.restore(sess, tf.train.latest_checkpoint('run1/'))
graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")

# Create the feed_dict that is required to calculate y_pred 
feed_dict_testing = {x: images, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)

# Output the predictions
with open(sys.argv[3],'w') as of:
    i=0
    for row in result:
        c=0
        max_p=0
        j=0
        for p in row:
            if p>max_p:
                max_p=p
                c=j
            j+=1
        print(files[i]+"\t"+classes[c])
        of.write(files[i]+"\t"+str(row)+"\t"+classes[c]+"\n")
        i+=1

