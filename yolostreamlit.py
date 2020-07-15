import streamlit as st
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from PIL import Image


def class_name():
    classes = []
    with open(r"C:\Users\movies\Desktop\python\new modules\yolov3_classname.txt","r") as f:
        classes = [i.strip() for i in f.readlines()]
        return classes

    

@st.cache
def yolo_detect(image_path):
    try:
        
        # load yolo
        weight_path =r"C:\Users\movies\Desktop\python\new modules\yolov3.weights"
        config_path =r"C:\Users\movies\Desktop\python\new modules\yolov3.cfg"
        net = cv2.dnn.readNet(weight_path , config_path)

        # creating a file for the yolo class name
        classes = class_name()
        
        #layer_names = net.getLayerNames()
        #outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
        

        # pickle load
        model_path = r"C:\Users\movies\Desktop\python\ml streamlit\yolo_model_pickle.pickle"

        outputlayers = pickle.load( open(model_path ,"rb"))
        
        # loading the image

        img_path=image_path
        img = cv2.imread(img_path)

        

        # resize an image
        img = cv2.resize(img ,None, fx=0.4, fy=0.3)

        print( img.shape )

        height, width, channels = img.shape

        # detecting the objects

        blob = cv2.dnn.blobFromImage(img , 0.00392,(416,416),(0,0,0),True,crop = False)


        # send the picture to the neural network

        net.setInput(blob)

        outs = net.forward(outputlayers)


        colors= np.random.uniform(0,255,size=(len(classes),3))



        #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids=[]
        confidences=[]
        boxes=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    #onject detected
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                
                    #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    
                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)


        font = cv2.FONT_HERSHEY_PLAIN
        #detected class in dicts

        detect_label ={}
        # create the dataframe
        object_count = []
        object_accuracy =[]
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img,(x,y),(x+w,y+h),color,4)
                cv2.putText(img,label,(x,y+30),font,1,(255,255,255),2)
                
                confy = max(confidences*1000000)
                print(confy)
                max_confy = str(confy).replace(str(confy)[4:],"%").replace(str(confy)[0],"")

                detect_label[label] = max_confy

                object_count.append(label)
                object_accuracy.append(max_confy)
                
                cv2.putText(img,(str(confy).replace(str(confy)[4:],"%").replace(str(confy)[0:2],"")),(x+45,y-10),font,1,(255,255,255),2)


        #plt.imshow(img)
        #cv2.imshow("Image",img)

        #cv2.imwrite(r"C:\Users\movies\Desktop\python\newyolo_result2_1.jpg",img)
        
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # converting the rgb color
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        
        return img,object_count,object_accuracy
    except:
        st.error("there was problem check the paths")
        return "error",0,1





if True:

    # style code
    f = """
  
            body {
            background-color : #c3015c;
            }
            .stButton>button  {
    display: inline-block;
    padding: .75rem 1.25rem;
    border-radius: 10rem;
    color: black;
    text-transform: uppercase;
    font-size: 1rem;
    letter-spacing: .15rem;
    transition: all .3s;
    position: relative;
    overflow: hidden;
    z-index: 1;
    &:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: black;
        border-radius: 10rem;
        z-index: -2;
    }
    &:before {
        content: 'press';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 0%;
        height: 100%;
        background-color: darken($color, 15%);
        transition: all .3s;
        border-radius: 10rem;
        z-index: -1;
    }
    &:hover {
        color: #fff;
        &:before {
            width: 100%;
        }
    }
}



        """

    # markdown
    st.markdown(f'<style>{f}</style>' , unsafe_allow_html = True)
    
    st.write(" # Basic testing")
    st.write("# Yolo Object Detection")
    st.write("""## Enter the **image path** see the result of the detected image with **detected object name**""")
    
    # sidebars

    st.sidebar.header(" About")
    st.sidebar.text(" yolo model for object detection")
    st.sidebar.text(" it can detect the object of class name")
    st.sidebar.text(" if it doesn't work or any queries \n contact me: jawaharlakshmanan6@gmail.com")
    classes = class_name()
    for i in classes:
        st.sidebar.text(i)
        
    # picture upload

    path = st.text_input("  Enter the path of the image to upload")
    
    
    if not(path):
        st.warning("Please enter the path or check the path")
    # button click

    if st.button("pressed"):
        
        with st.spinner("please wait loading..."):
            img ,df1,df2= yolo_detect(path)

    

        if df1==0:
            pass

        else:
            st.image(img,caption = "detected image",use_column_width = True)
            # creating a dataframe

            pd.Series(df1)

            s1 = pd.Series(df1)
            s2 = pd.Series(df2)
            data_dict = {
                        "Object name":s1, "Accuracy":s2, "status":"detected"
                }
            df = pd.DataFrame(data = data_dict)
        
            st.write(df)




