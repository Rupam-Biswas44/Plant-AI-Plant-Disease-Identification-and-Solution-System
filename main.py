import openai
import streamlit as st
import tensorflow as tf
import numpy as np
import base64
import time

import numpy as np

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)





def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

with st.popover("Open popover"):
    st.markdown("Hello 👋")
    name = st.text_input("What's your name?")

st.write("Your name:", name)
with st.sidebar:
    
    

    with st.spinner("Loading..."):
        time.sleep(3)
    st.success("Done!")
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #0000;
    }
</style>
""", unsafe_allow_html=True)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('aiplant.png')


    
st.sidebar.title("Browse Me")
app_mode = st.sidebar.selectbox("Select from below :",["PlantAI-Home Page","Model Details","Disease Identification","Help from AI","Contact Me"])

if(app_mode=="PlantAI-Home Page"):
    st.title(":blue[_PLANT-AI_] : PLANT DISEASE DETECTION")
    st.divider()
    st.subheader('Welcome to the PLANT-AI : PLANT DISEASE DETECTION')
    st.subheader('Motivation Behind this APP:', divider='rainbow')
    st.markdown("""Early plant disease recognition is crucial for farmers to ensure good crop production as it allows for timely intervention and management. Identifying diseases at their onset enables farmers to implement appropriate control measures promptly, preventing the spread and escalation of the disease throughout the crop. Timely action can include targeted application of fungicides, adjustment of irrigation practices, or even removal of infected plants to prevent further contamination. By detecting diseases early, farmers can minimize yield losses, preserve crop quality, and ultimately safeguard their livelihoods against potential economic losses caused by unchecked disease outbreaks. Thus, early disease recognition serves as a proactive approach in maintaining optimal crop health and maximizing agricultural productivity.
                """)

    

    st.subheader('A Quick Guide:', divider='rainbow')

    st.markdown("""
    - **Upload Image:** Upload an Image of a plant leaf from Gallery of your device in **Disease Recognition** page.
    - **Click Live picture From Device Camera:** If you want to examine a plant leaf instantly , in **Disease Recognition** page click take photo.
    - **Analysis:** My system will analyze the image with Advance deep Learning algorithm and find out disease.
    - **Find Solution with AI:** My Open AI powered chat-Bot will provide probable solution of found disease and other question of you instantly .
    """)
    st.subheader('Speciallity of my Model:', divider='rainbow')
    st.markdown("""

    
    - This Model is Highly Accurate with almost 99 Percent Accuracy.
    - This App is easy to use and User friendly
    - A deep learning model is used which is very fast and efficient.
    """)


    
elif(app_mode=="Contact Me"):
    st.subheader('Contact Me', divider='rainbow')
    st.markdown("""
                - **AUTHOR:** Rupam Biswas
                - **Email Me:** rupambiswasbd44@gmail.com
                - **phon:** 9638356873
                - **Address:** Flat C-201,Shri Hari Residency,New CG Road,
                            Nigam Nagar,Chandkheda,Ahmedabad,Gujarat

                 """)

#About Project
elif(app_mode=="Model Details"):
    import streamlit as st
    st.markdown("<h1 style='text-align: center; color: white;'>About my DL Model</h1>", unsafe_allow_html=True)
    
    st.subheader('Dataset', divider='rainbow')
    st.markdown("""
                 
                Offline augmentation is used to recreate this dataset from the original dataset.You may access the original dataset from this github repository.
                This dataset, which is divided into 38 classes, includes over 87K rgb photos of both healthy and damaged crop leaves.
                The entire dataset is split up into training and validation sets in an 80/20 ratio while maintaining the directory structure. 
                Later on, a new directory with 33 test photos is made for prediction purposes.
                

                """)
    st.subheader('Content', divider='rainbow')
    st.markdown("""
                
                
                For training Purpose I have used almost 75000 images.
                For testing my model I have tested with almost 35 pictures and there I have got 98 percent accuracy
                I have validated the model with 20000 pictures     """)

    

#Prediction Page
elif(app_mode=="Help from AI"):
    st.header("HELP FROM AI",divider='rainbow')
    openai.api_key=st.secrets["OPEN_API_KEY"]

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
    
      


elif(app_mode=="Disease Identification",):
    
    st.header("Disease Identification",divider='rainbow')
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
    # To read image file buffer as a 3D uint8 tensor with TensorFlow:
        bytes_data = img_file_buffer.getvalue()
        img_tensor = tf.io.decode_image(bytes_data, channels=3)

    # Check the type of img_tensor:
    # Should output: <class 'tensorflow.python.framework.ops.EagerTensor'>
        st.write(type(img_tensor))

    # Check the shape of img_tensor:
    # Should output shape: (height, width, channels)
        st.write(img_tensor.shape)
    test_image = st.file_uploader("Choose an Image from your gellary:")
    if(st.button("See uploaded Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("See The condition of the plant")):
        st.snow()
        st.write("The identified disease is:")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("So.. We are predicting it's a {}".format(class_name[result_index]))




