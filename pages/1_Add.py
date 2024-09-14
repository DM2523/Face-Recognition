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

st.title('Add new person to the system.')
st.caption('Please upload image to continue...')

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
        if file_details['filetype'] in ( 'image/jpeg'):
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

#add to embeddings file
def add_new_person(new_person_image, new_person_name, embeddings_file):
    # Load existing embeddings
    embeddings = load_embeddings(embeddings_file)
    
    # Compute the new person's embedding
    new_person_embedding = get_embedding(new_person_image)
    
    # Add the new person's embedding to the existing embeddings
    embeddings[new_person_name] = new_person_embedding
    
    # Save the updated embeddings
    np.save(embeddings_file, embeddings)
    return new_person_name

#get new person name
def getPersonName():
    input = st.text_input("Person Name")
    return input

def main():
    Person_Name = getPersonName()
    image_file,file_details = upload_image()
    if image_file and Person_Name:
        prediction = False
        image  = Image.open(image_file)

        col1, col2 = st.columns(2)
        with col1:
            st.info('Preview of uploaded image')
            st.image(image_file,use_column_width=True)

        with col2:
            st.subheader('Check file details')
            st.json(file_details)

            button = st.button('Add')
            # image_arr = np.array(image)
            # st.write("Image array shape:", image_arr.shape)


            if button:
                with st.spinner('Adding to backend...'):
                    image_arr = np.array(image)
                    pred_img,person_boxes = model.predict(image_arr)
                    pred_img_obj_out = Image.fromarray(pred_img)
                    # st.image(pred_img_obj_out)
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
                        face_gray = face[0]
                        pred_img_obj = Image.fromarray(face_gray)
                        st.image(pred_img_obj)
                        recognized_person = add_new_person(face_gray,Person_Name, embeddings_file)
                        st.write(recognized_person)

            else:
                st.write("No Face Detected!")

if __name__ == '__main__':
    main()

