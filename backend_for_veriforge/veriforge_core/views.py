# from django.shortcuts import render
# from django.contrib.auth.decorators import login_required
# from django.core.files.storage import default_storage
# from django.conf import settings
# import os
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# from veriforge_core.utils import ela_image
# # from . import utils

# # Load the model once when the module is imported
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.h5')
# model = tf.keras.models.load_model(MODEL_PATH)

# input_shape = (224, 224, 3)

# def home(request):
#     if request.method == 'POST' and request.FILES['image']:
#         uploaded_image = request.FILES['image']

#         # Ensure image is in a compatible format
#         image = Image.open(uploaded_image).convert('RGB')
#         # image = image.resize((224, 224, 3))  # VGG16 input size

#         # img = Image.open(img_path)
#         image = image.resize(input_shape[:2])

#         # Apply ELA preprocessing
#         ela_image_processed = ela_image(image)
#         img_array = np.array(ela_image_processed, dtype=np.float32) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         # # Load model
#         # model_path = os.path.join(os.path.dirname(__file__), 'model', 'VGG16.h5')
#         # model = tf.keras.models.load_model(model_path)

#         # Make prediction
#         prediction = model.predict(img_array)
#         print("prediction:", prediction)
#         # tampered = prediction[0][0] > 0.5  # Assuming binary output, adjust for your model

#         return render(request, 'index.html', {
#             'message': 'Prediction complete!',
#             'is_tampered': 'Tampered' if prediction[0][0] > 0.5 else 'Authentic',
#         })

#     return render(request, 'index.html')



from django.shortcuts import render
import numpy as np
from PIL import Image
from veriforge_core.utils import ela_image
from huggingface_hub import hf_hub_download
import tensorflow as tf

# Hugging Face model repo and file path
MODEL_REPO = "pateltrushit1710/veriforge"
MODEL_FILENAME = "model.h5"

# Download the .h5 file from Hugging Face
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

# Load the model
model = tf.keras.models.load_model(model_path)

input_shape = (224, 224, 3)

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']

        # Ensure image is in a compatible format
        image = Image.open(uploaded_image).convert('RGB')
        image = image.resize(input_shape[:2])

        # Apply ELA preprocessing
        ela_image_processed = ela_image(image)
        img_array = np.array(ela_image_processed, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        print("Prediction:", prediction)

        return render(request, 'index.html', {
            'message': 'Prediction complete!',
            'is_tampered': 'Tampered' if prediction[0][0] > 0.5 else 'Authentic',
        })

    return render(request, 'index.html')
