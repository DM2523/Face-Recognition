import streamlit as st
from Inference import YOLO_PRED
from PIL import Image
import numpy as np
import cv2
from model import FaceNetModel
import torch
from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1



st.set_page_config(page_title = 'Image Detection',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title('Welcome to Facial Recognition App')
st.caption('Please upload image to get started.')

def load_embeddings(file_path):
    embeddings = np.load(file_path, allow_pickle=True).item()
    return embeddings


#loading detection model
with st.spinner('Loading model...'):
    model = YOLO_PRED(onn_model='./yolov8s.onnx',data_yaml='./data.yaml')
    faceDetector = cv2.CascadeClassifier()
    # Load the pre-trained Haar Cascade for face detection
    faceDetector.load(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #import embedding model
    # FaceNet = FaceNetModel()
    # FaceNet.load_state_dict(torch.load('CustomFaceNetEncoder.pth', weights_only=True))
    FaceNet = InceptionResnetV1(pretrained='vggface2').eval()

    #load embeddings
    embeddings_file = './employee_embeddings.npy'
    embeddings = load_embeddings(embeddings_file)
    # st.balloons()


#input image
def upload_image():
    # Upload image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        file_details = {
            "filename": image_file.name,
            "filetype": image_file.type,
            "filesize": "{:,.2f}MB".format(image_file.size / 1024**2)
        }

        # st.write(file_details)

        # Validate File
        if file_details['filetype'] in ('image/jpeg'):
            st.success('VALID FILE.')
            return image_file, file_details
        else:
            st.error('INVALID FILE. Please upload a valid image file.')
            return None, None
    return None, None

#cropping ROIs
def getROIs(boxes,image):
    image_list = []

    for i in boxes:
        [(x1,y1),(x2,y2)] = i
        sub_image = image[y1:y2, x1:x2, :]  
        image_list.append(sub_image)
    
    return image_list

#getting faces
def getFaces(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #Resizing
    # faceDetector = cv2.CascadeClassifier()
    # # Load the pre-trained Haar Cascade for face detection
    # faceDetector.load(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detections = faceDetector.detectMultiScale(image_gray)
    face_list = []
    for i in detections:
        x1,y1,w,h = i
        sub_image = image[y1:y1+h, x1:x1+w,:]
        face_list.append(sub_image)
    return face_list

#calculate embeddings
def get_embedding(image):
    image = cv2.resize(image,(160,160),cv2.INTER_CUBIC)
    image = torch.Tensor(image)
    image=image.permute(2,0,1)
    image = image
    image = image.unsqueeze(0)
    # print(image.shape)# Add batch dimension

    with torch.no_grad():

        embedding = FaceNet(image)  # Get embedding from the model

    return embedding.squeeze().cpu().numpy() 

#Get recognition
def recognize_face(input_image, embeddings, threshold=0.5):
    query_embedding = get_embedding(input_image)
    
    min_distance = float('inf')
    recognized_name = 'Unknown'
    records={}
    
    for name, stored_embedding in embeddings.items():
        distance = cosine(query_embedding, stored_embedding)
        records[name]=distance
        if distance < min_distance:
            min_distance = distance
            recognized_name = name
    
    if min_distance > threshold:
        recognized_name = 'Unknown'
    
    return recognized_name,records

def main():
    image_file,file_details = upload_image()
    if image_file:
        prediction = False
        image  = Image.open(image_file)

        col1, col2 = st.columns(2)
        with col1:
            st.info('Preview of uploaded image')
            st.image(image_file,use_column_width=True)

        with col2:
            st.subheader('Check file details')
            st.json(file_details)

            button = st.button('Recognise')
            # image_arr = np.array(image)
            # st.write("Image array shape:", image_arr.shape)


            if button:
                with st.spinner('Recognising...'):
                    image_arr = np.array(image)
                    # Create the sharpening kernel 
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
  
                    # Sharpen the image 
                    image_arr = cv2.filter2D(image_arr, -1, kernel) 
                    pred_img,person_boxes = model.predict(image_arr)
                    pred_img_obj = Image.fromarray(pred_img)
                    # st.image(pred_img_obj)
                    if(len(person_boxes)!=0):
                        prediction = True
                        person_list = getROIs(person_boxes,image_arr)
                    else:
                        st.write('No person detected.')

        if prediction:
            # Add border to the image
            st.subheader('Output')
            face_detected=False
            face_list = [getFaces(person) for person in person_list]
            if(len(face_list)!=0):
                face_detected=True
            if(face_detected):
                for face in face_list:
                    if(len(face)):
                        # face_gray = cv2.cvtColor(face[0],cv2.COLOR_BGR2GRAY)
                        pred_img_obj = Image.fromarray(face[0])
                        st.image(pred_img_obj)
                        recognized_person,records = recognize_face(face[0], embeddings)
                        st.write(recognized_person)
                        # st.json(records)

            else:
                st.write("No Face Detected!")

if __name__ == '__main__':
    main()
