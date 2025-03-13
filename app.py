import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from retinaface import RetinaFace
from PIL import Image
import random
import time
import json
import os
from functools import lru_cache
import pandas as pd
import plotly.express as px
import threading
import concurrent.futures

import tensorflow as tf
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def check_gpu_availability():
    gpu_available = False
    gpu_info = ""
    
    tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    
    torch_gpu = False
    if TORCH_AVAILABLE:
        torch_gpu = torch.cuda.is_available()
    
    gpu_available = tf_gpu or torch_gpu
    
    if gpu_available:
        if tf_gpu:
            gpu_info += f"TensorFlow GPU: {tf.config.list_physical_devices('GPU')}\n"
        if torch_gpu:
            gpu_info += f"PyTorch GPU: {torch.cuda.get_device_name(0)}\n"
            
    return gpu_available, gpu_info

def configure_gpu():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            st.sidebar.success("GPU configured for optimal performance")
    except Exception as e:
        st.sidebar.warning(f"Could not configure GPU: {e}")

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

def save_user_preferences(prefs):
    try:
        with open(USER_PREF_PATH, "w") as f:
            json.dump(prefs, f, cls=NumpyJSONEncoder)
    except Exception as e:
        st.error(f"Failed to save preferences: {e}")

GPU_AVAILABLE, GPU_INFO = check_gpu_availability()
if GPU_AVAILABLE:
    configure_gpu()
    
if 'model_cache' not in st.session_state:
    st.session_state['model_cache'] = {}

@lru_cache(maxsize=4)
def load_face_model(model_name, use_gpu=False):
    if model_name in st.session_state['model_cache']:
        return st.session_state['model_cache'][model_name]
    
    if use_gpu and GPU_AVAILABLE:
        if model_name == "VGG-Face":
            pass
    
    st.session_state['model_cache'][model_name] = model_name
    return model_name

def preprocess_image(img_array):
    height, width = img_array.shape[:2]
    max_dimension = 640 
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        img_array = cv2.resize(img_array, (new_width, new_height))
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    
    brightened = cv2.convertScaleAbs(img_array, alpha=1.5, beta=30)
    
    return {"original": img_array, "enhanced": enhanced, "brightened": brightened}

def detect_faces_with_fallback(img_array):
    faces = None
    detection_method = "none"

    preprocessed = preprocess_image(img_array)

    try:
        faces = RetinaFace.detect_faces(preprocessed["original"])
        if faces:
            detection_method = "retinaface_original"
            return faces, detection_method
    except Exception:
        pass

    try:
        faces = RetinaFace.detect_faces(preprocessed["enhanced"])
        if faces:
            detection_method = "retinaface_enhanced"
            return faces, detection_method
    except Exception:
        pass

    try:
        faces = RetinaFace.detect_faces(preprocessed["brightened"])
        if faces:
            detection_method = "retinaface_brightened"
            return faces, detection_method
    except Exception:
        pass

    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        opencv_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(opencv_faces) > 0:
            faces = {}
            for i, (x, y, w, h) in enumerate(opencv_faces):
                faces[f"face_{i+1}"] = {"facial_area": (x, y, w, h), "score": 0.9}
            detection_method = "opencv_haar"
            return faces, detection_method
    except Exception:
        pass

    try:
        result = DeepFace.extract_faces(
            img_path=img_array,
            enforce_detection=False,
            detector_backend="opencv",
            align=True,
        )

        if result and len(result) > 0:
            faces = {}
            for i, face_data in enumerate(result):
                facial_area = face_data.get("facial_area", {})
                x = facial_area.get("x", 0)
                y = facial_area.get("y", 0)
                w = facial_area.get("w", 0)
                h = facial_area.get("h", 0)
                faces[f"face_{i+1}"] = {"facial_area": (x, y, w, h), "score": 0.8}
            detection_method = "deepface_detector"
            return faces, detection_method
    except Exception:
        pass

    if not faces:
        try:
            from deepface.detectors import MTCNN

            detector = MTCNN()
            dets = detector.detect_faces(img_array)
            if len(dets) > 0:
                faces = {}
                detection_method = "mtcnn_fallback"
                for i, d in enumerate(dets):
                    x, y, w, h = d["box"]
                    faces[f"face_{i+1}"] = {
                        "facial_area": (x, y, w, h),
                        "score": d.get("confidence", 0.8),
                    }
                return faces, detection_method
        except Exception:
            pass

    return None, detection_method

def analyze_with_models(img_array):
    results = []
    
    models_to_try = ["VGG-Face", "Facenet", "DeepID"] if GPU_AVAILABLE else ["VGG-Face", "Facenet"]
    
    analyze_kwargs = {
        "actions": ["emotion"],  
        "enforce_detection": False,
        "detector_backend": "skip",  
        "prog_bar": False
    }
    
    if GPU_AVAILABLE:
        analyze_kwargs["use_gpu"] = True
    
    for model in models_to_try:
        try:
            model_obj = load_face_model(model, use_gpu=GPU_AVAILABLE)
            
            result = DeepFace.analyze(
                img_path=img_array,
                **analyze_kwargs
            )
            
            if isinstance(result, list):
                result = result[0]
            
            if "emotion" in result:
                for key in result["emotion"]:
                    if isinstance(result["emotion"][key], (np.floating, np.integer)):
                        result["emotion"][key] = float(result["emotion"][key])
            
            results.append(result)
            
            if len(results) >= 2:
                break
                
        except Exception as e:
            pass
    
    return results

def weighted_average_results(results):
    
    if not results:
        return None
        
    avg_result = results[0].copy()
    weights = [1.0] * len(results)

    for i, res in enumerate(results[1:], start=1):
        if "emotion" in res and "emotion" in avg_result:
            for key in avg_result["emotion"]:
                if key in res["emotion"]:
                    avg_result["emotion"][key] += float(res["emotion"][key]) * weights[i]

    total_weight = sum(weights)
    if "emotion" in avg_result:
        for key in avg_result["emotion"]:
            avg_result["emotion"][key] = float(avg_result["emotion"][key]) / total_weight
        
        dominant_key = max(avg_result["emotion"], key=avg_result["emotion"].get)
        avg_result["dominant_emotion"] = dominant_key

    return avg_result

def analyze_emotion_with_models(img_array):
    max_dimension = 320 
    height, width = img_array.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        small_img = cv2.resize(img_array, (int(width * scale), int(height * scale)))
    else:
        small_img = img_array.copy()
    
    faces, detection_method = detect_faces_with_fallback(small_img)

    if not faces:
        return (
            None,
            0,
            "No face was detected. Please use better lighting or adjust your face position.",
        )

    try:
        if isinstance(faces, dict) and len(faces) > 0:
            face_key = list(faces.keys())[0]
            face = faces[face_key]
            
            x, y, w, h = face["facial_area"]
            
            if max(height, width) > max_dimension:
                scale_factor = max(height, width) / max_dimension
                x = int(x * scale_factor)
                y = int(y * scale_factor)
                w = int(w * scale_factor)
                h = int(h * scale_factor)
            
            x, y = max(0, x), max(0, y)
            w = min(w, img_array.shape[1] - x)
            h = min(h, img_array.shape[0] - y)

            if w > 0 and h > 0:
                face_img = img_array[y:y+h, x:x+w]
            else:
                face_img = img_array
        else:
            face_img = img_array
            
        results = analyze_with_models(face_img)
        
        if not results:
            return (
                None,
                0,
                "A face was detected, but we couldn't analyze the emotion. Please try a different angle or change your expression.",
            )
            
        result_dict = weighted_average_results(results)
        if not result_dict or "emotion" not in result_dict:
            return (
                None,
                0,
                "We found your face, but couldn't process the emotion data. Please try again with better lighting.",
            )
            
        dominant_emotion = result_dict["dominant_emotion"]
        confidence = result_dict["emotion"][dominant_emotion] * 100
        
        confidence = float(confidence)
        
        if confidence < 35:
            return (
                None,
                confidence,
                "We found your face, but the confidence was too low to determine your mood. Please try a different angle or select your mood manually."
            )
            
        emotion_results = {
            "emotions": {k: float(v) for k, v in result_dict["emotion"].items()},
            "dominant_emotion": dominant_emotion
        }
        
        return emotion_results, confidence, detection_method
            
    except Exception as e:
        st.error(f"Error in emotion analysis: {str(e)}")
        return (
            None,
            0,
            "An error occurred during emotion analysis. Please try again."
        )

st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded",
)

USER_PREF_PATH = os.path.join(os.path.dirname(__file__), "user_preferences.json")

hindi_music_recommendations = {
    "happy": [
        {
            "title": "Badtameez Dil – Yeh Jawaani Hai Deewani",
            "url": "https://www.youtube.com/watch?v=II2EO3Nw4m0",
        },
        {
            "title": "Gallan Goodiyaan – Dil Dhadakne Do",
            "url": "https://www.youtube.com/watch?v=jCEdTq3j-0U",
        },
        {
            "title": "Nagada Sang Dhol – Goliyon Ki Raasleela Ram-Leela",
            "url": "https://www.youtube.com/watch?v=3X7x4Ye-tqo",
        },
        {
            "title": "London Thumakda – Queen",
            "url": "https://www.youtube.com/watch?v=udra3Mfw2oo",
        },
        {
            "title": "Balam Pichkari – Yeh Jawaani Hai Deewani",
            "url": "https://www.youtube.com/watch?v=0WtRNGubWGA",
        },
        {
            "title": "Tune Maari Entriyaan – Gunday",
            "url": "https://www.youtube.com/watch?v=2I3NgxDAiqE",
        },
        {
            "title": "Kar Gayi Chull – Kapoor & Sons",
            "url": "https://www.youtube.com/watch?v=NTHz9ephYTw",
        },
        {
            "title": "Ghungroo – War",
            "url": "https://www.youtube.com/watch?v=qFkNATtc3mc",
        },
        {
            "title": "Aankh Marey – Simmba",
            "url": "https://www.youtube.com/watch?v=O6OI2-p-gC4",
        },
        {
            "title": "Morni Banke – Badhaai Ho",
            "url": "https://www.youtube.com/watch?v=h-v5tHtL_cA",
        },
        {
            "title": "Dil Dhadakne Do – Title Track",
            "url": "https://www.youtube.com/watch?v=R5jQ8VudZbA",
        },
        {
            "title": "Nashe Si Chadh Gayi – Befikre",
            "url": "https://www.youtube.com/watch?v=Wd2B8OAOc9c",
        },
        {
            "title": "Tamma Tamma Again – Badrinath Ki Dulhania",
            "url": "https://www.youtube.com/watch?v=EEX_XM6SxmY",
        },
        {
            "title": "Proper Patola – Namaste England",
            "url": "https://www.youtube.com/watch?v=Y1O_aIV1FNM",
        },
        {
            "title": "Abhi Toh Party Shuru Hui Hai – Khoobsurat",
            "url": "https://www.youtube.com/watch?v=8LZgzAZ2lpQ",
        },
    ],
    "sad": [
        {
            "title": "Channa Mereya – Ae Dil Hai Mushkil",
            "url": "https://www.youtube.com/watch?v=284Ov7ysmfA",
        },
        {
            "title": "Tum Hi Ho – Aashiqui 2",
            "url": "https://www.youtube.com/watch?v=Umqb9KENgmk",
        },
        {
            "title": "Luka Chuppi – Rang De Basanti",
            "url": "https://www.youtube.com/watch?v=_ikZtcgAMxo",
        },
        {
            "title": "Agar Tum Saath Ho – Tamasha",
            "url": "https://www.youtube.com/watch?v=sK7riqg2mr4",
        },
        {
            "title": "Judaai – Badlapur",
            "url": "https://www.youtube.com/watch?v=zPpNZFgSzDo",
        },
        {
            "title": "Kabhi Alvida Naa Kehna – KANK",
            "url": "https://www.youtube.com/watch?v=O8fIwHfZz2E",
        },
        {
            "title": "Main Dhoondne Ko Zamaane Mein – Heartless",
            "url": "https://www.youtube.com/watch?v=5wqoxs9zG3w",
        },
        {
            "title": "Tujhe Bhula Diya – Anjaana Anjaani",
            "url": "https://www.youtube.com/watch?v=_SK9K58Olqo",
        },
        {
            "title": "Phir Le Aaya Dil – Barfi!",
            "url": "https://www.youtube.com/watch?v=ntC3sO-VeJY",
        },
        {
            "title": "Bhula Dena – Aashiqui 2",
            "url": "https://www.youtube.com/watch?v=j4OVQJ6-R1U",
        },
        {
            "title": "Kabira (Encore) – Yeh Jawaani Hai Deewani",
            "url": "https://www.youtube.com/watch?v=jHNNMj5bNQw",
        },
        {
            "title": "Tera Ban Jaunga – Kabir Singh",
            "url": "https://www.youtube.com/watch?v=mQiiw7uRngA",
        },
        {
            "title": "Humsafar – Badrinath Ki Dulhania",
            "url": "https://www.youtube.com/watch?v=8v-TWxPWIWc",
        },
        {
            "title": "Agar Tum Mil Jao – Zeher",
            "url": "https://www.youtube.com/watch?v=C8tMQzcPXOY",
        },
        {
            "title": "Dil Ke Paas – Wajah Tum Ho",
            "url": "https://www.youtube.com/watch?v=FOLUdLQBPuE",
        },
    ],
    "angry": [
        {
            "title": "Challa – Jab Tak Hai Jaan",
            "url": "https://www.youtube.com/watch?v=9a4izd3Rvdw",
        },
        {
            "title": "Brothers Anthem – Brothers",
            "url": "https://www.youtube.com/watch?v=IjBAgWKW12Y",
        },
        {
            "title": "Sultan – Sultan",
            "url": "https://www.youtube.com/watch?v=RYvUMglNznM",
        },
        {
            "title": "Bulleya – Ae Dil Hai Mushkil",
            "url": "https://www.youtube.com/watch?v=hXh35CtnSyU",
        },
        {
            "title": "Sadda Haq – Rockstar",
            "url": "https://www.youtube.com/watch?v=p9DQINKZxWE",
        },
        {
            "title": "Jee Karda – Badlapur",
            "url": "https://www.youtube.com/watch?v=BN45QQ7R92M",
        },
        {
            "title": "Dhan Te Nan – Kaminey",
            "url": "https://www.youtube.com/watch?v=m9RdKcnUvFU",
        },
        {
            "title": "Bhaag DK Bose – Delhi Belly",
            "url": "https://www.youtube.com/watch?v=IQEDu8SPHao",
        },
        {
            "title": "Sher Aaya Sher – Gully Boy",
            "url": "https://www.youtube.com/watch?v=hejXc_FSYb8",
        },
        {
            "title": "Aala Re Aala – Simmba",
            "url": "https://www.youtube.com/watch?v=2wbVxHlOepM",
        },
        {
            "title": "Zinda – Bhaag Milkha Bhaag",
            "url": "https://www.youtube.com/watch?v=RLzC55ai0eo",
        },
        {
            "title": "Mardaani – Mardaani",
            "url": "https://www.youtube.com/watch?v=C1QOVnH0bKY",
        },
        {
            "title": "Jai Ho – Slumdog Millionaire",
            "url": "https://www.youtube.com/watch?v=Yc5OyXmHD0w",
        },
        {
            "title": "Malhari – Bajirao Mastani",
            "url": "https://www.youtube.com/watch?v=l_MyUGq7pgs",
        },
        {
            "title": "Apna Time Aayega – Gully Boy",
            "url": "https://www.youtube.com/watch?v=SlHnlxDt2TQ",
        },
    ],
    "fear": [
        {
            "title": "Darr Ke Aage Jeet Hai – Mountain Dew",
            "url": "https://www.youtube.com/watch?v=xT7E-n1t3vI",
        },
        {
            "title": "Hai Ram – Sarkar Raj",
            "url": "https://www.youtube.com/watch?v=zDQUylzavMQ",
        },
        {
            "title": "Aaj Phir Jeene Ki Tamanna Hai – Guide",
            "url": "https://www.youtube.com/watch?v=2LG8LwEVlJE",
        },
        {
            "title": "Main Hoon Don – Don",
            "url": "https://www.youtube.com/watch?v=xvNgjtgXOVo",
        },
        {
            "title": "Khalbali – Rang De Basanti",
            "url": "https://www.youtube.com/watch?v=Yd-ngSyDZss",
        },
        {
            "title": "Bhoot Hoon Main – Bhoot",
            "url": "https://www.youtube.com/watch?v=JNV4To5uzKA",
        },
        {
            "title": "Bhool Bhulaiyaa – Title Track",
            "url": "https://www.youtube.com/watch?v=eN6AYHAT8UM",
        },
        {
            "title": "Darr – Title Track",
            "url": "https://www.youtube.com/watch?v=BTAXAc1bJh8",
        },
        {
            "title": "Pari – Title Track",
            "url": "https://www.youtube.com/watch?v=ZwyKOXwJsC0",
        },
        {
            "title": "Bol Na Halke Halke – Jhoom Barabar Jhoom",
            "url": "https://www.youtube.com/watch?v=S9LZjGGeedw",
        },
        {
            "title": "Raat Ka Nasha – Asoka",
            "url": "https://www.youtube.com/watch?v=OjaFNUA-UFE",
        },
        {
            "title": "Phir Se Ud Chala – Rockstar",
            "url": "https://www.youtube.com/watch?v=2mWaqsC3U7k",
        },
        {
            "title": "Roobaroo – Rang De Basanti",
            "url": "https://www.youtube.com/watch?v=BrfRB6aTZlM",
        },
        {
            "title": "Khamoshiyan – Khamoshiyan",
            "url": "https://www.youtube.com/watch?v=FXiaIH49oAU",
        },
        {
            "title": "Tum Ho Toh – Rock On!!",
            "url": "https://www.youtube.com/watch?v=hCsY8T0uBGA",
        },
    ],
    "neutral": [
        {
            "title": "Kabira – Yeh Jawaani Hai Deewani",
            "url": "https://www.youtube.com/watch?v=jHNNMj5bNQw",
        },
        {
            "title": "Mitwa – Kabhi Alvida Naa Kehna",
            "url": "https://www.youtube.com/watch?v=Cv86br9MSBc",
        },
        {
            "title": "Kun Faya Kun – Rockstar",
            "url": "https://www.youtube.com/watch?v=T94PHkuydcw",
        },
        {
            "title": "Tum Se Hi – Jab We Met",
            "url": "https://www.youtube.com/watch?v=mt9xg0mmt28",
        },
        {
            "title": "Iktara – Wake Up Sid",
            "url": "https://www.youtube.com/watch?v=fSS_R91Nimw",
        },
        {
            "title": "Nazm Nazm – Bareilly Ki Barfi",
            "url": "https://www.youtube.com/watch?v=DK_UsATwoxI",
        },
        {
            "title": "Ae Dil Hai Mushkil – ADHM",
            "url": "https://www.youtube.com/watch?v=6FURuLYrR_Q",
        },
        {
            "title": "Raabta – Agent Vinod",
            "url": "https://www.youtube.com/watch?v=zAU_rsoS5ok",
        },
        {
            "title": "Tera Hone Laga Hoon – Ajab Prem Ki Ghazab Kahani",
            "url": "https://www.youtube.com/watch?v=K0IvuwrSFaI",
        },
        {
            "title": "Safarnama – Tamasha",
            "url": "https://www.youtube.com/watch?v=zLv0V_19L-A",
        },
        {
            "title": "Pehli Nazar Mein – Race",
            "url": "https://www.youtube.com/watch?v=BadBAMnPX0I",
        },
        {
            "title": "Saibo – Shor in the City",
            "url": "https://www.youtube.com/watch?v=zXLgYBSdv74",
        },
        {
            "title": "O Re Piya – Aaja Nachle",
            "url": "https://www.youtube.com/watch?v=iv7lcUkFVSc",
        },
        {
            "title": "Khairiyat – Chhichhore",
            "url": "https://www.youtube.com/watch?v=hoNb6HuNmU0",
        },
        {
            "title": "Manwa Laage – Happy New Year",
            "url": "https://www.youtube.com/watch?v=d8IT-16kA8M",
        },
    ],
    "disgust": [
        {
            "title": "Beedi – Omkara",
            "url": "https://www.youtube.com/watch?v=XLJCtZKQGrY",
        },
        {
            "title": "Emotional Atyachaar – Dev D",
            "url": "https://www.youtube.com/watch?v=Vng5mg0iY0k",
        },
        {
            "title": "Gandi Baat – R... Rajkumar",
            "url": "https://www.youtube.com/watch?v=vvLBXO4MnKQ",
        },
        {
            "title": "Bluffmaster – Bluffmaster",
            "url": "https://www.youtube.com/watch?v=t5UzO4gOYKc",
        },
        {
            "title": "Dhoom Machale – Dhoom",
            "url": "https://www.youtube.com/watch?v=ymk2_5a2V8g",
        },
        {
            "title": "Zor Ka Jhatka – Action Replayy",
            "url": "https://www.youtube.com/watch?v=UZV8Yb4hVgU",
        },
        {
            "title": "Genda Phool – Delhi-6",
            "url": "https://www.youtube.com/watch?v=f0CvPQ3l-Xg",
        },
        {
            "title": "Dum Maro Dum – Hare Rama Hare Krishna",
            "url": "https://www.youtube.com/watch?v=BZNT_Y-mAkE",
        },
        {
            "title": "Chaar Botal Vodka – Ragini MMS 2",
            "url": "https://www.youtube.com/watch?v=x8F5dz8kv1w",
        },
        {
            "title": "Kamli – Dhoom 3",
            "url": "https://www.youtube.com/watch?v=C8kSrkz8Hz8",
        },
        {
            "title": "Munni Badnaam Hui – Dabangg",
            "url": "https://www.youtube.com/watch?v=Jn5hsfbhWx4",
        },
        {
            "title": "Sheila Ki Jawani – Tees Maar Khan",
            "url": "https://www.youtube.com/watch?v=ZTmF2v59CtI",
        },
        {
            "title": "Baby Doll – Ragini MMS 2",
            "url": "https://www.youtube.com/watch?v=yP9KiFTyBks",
        },
        {
            "title": "Oo Antava – Pushpa",
            "url": "https://www.youtube.com/watch?v=kyNdRJR_NRs",
        },
        {
            "title": "Laila Main Laila – Raees",
            "url": "https://www.youtube.com/watch?v=fMW7ze7-Gik",
        },
    ],
    "surprise": [
        {
            "title": "Kala Chashma – Baar Baar Dekho",
            "url": "https://www.youtube.com/watch?v=k4yXQkG2s1E",
        },
        {
            "title": "Matargashti – Tamasha",
            "url": "https://www.youtube.com/watch?v=6vKucgAeF_Q",
        },
        {
            "title": "Sooraj Dooba Hain – Roy",
            "url": "https://www.youtube.com/watch?v=nJZcbidTutE",
        },
        {
            "title": "Ghagra – Yeh Jawaani Hai Deewani",
            "url": "https://www.youtube.com/watch?v=caoGNx1LF2Q",
        },
        {
            "title": "Ishq Shava – Jab Tak Hai Jaan",
            "url": "https://www.youtube.com/watch?v=2kH4fLrwc0A",
        },
        {
            "title": "Naina – Dangal",
            "url": "https://www.youtube.com/watch?v=BTtx6HMzunQ",
        },
        {
            "title": "Haanikaarak Bapu – Dangal",
            "url": "https://www.youtube.com/watch?v=KyZZrClK0rM",
        },
        {
            "title": "Malhari – Bajirao Mastani",
            "url": "https://www.youtube.com/watch?v=l_MyUGq7pgs",
        },
        {
            "title": "Dilbar – Satyameva Jayate",
            "url": "https://www.youtube.com/watch?v=JFcgOboQZ08",
        },
        {
            "title": "Bom Diggy Diggy – Sonu Ke Titu Ki Sweety",
            "url": "https://www.youtube.com/watch?v=yIIGQB6EMAM",
        },
        {
            "title": "Cutiepie – Ae Dil Hai Mushkil",
            "url": "https://www.youtube.com/watch?v=f6vY6tYvKGA",
        },
        {
            "title": "Coca Cola – Luka Chuppi",
            "url": "https://www.youtube.com/watch?v=_cPHiwPqbqo",
        },
        {
            "title": "Sweety Tera Drama – Bareilly Ki Barfi",
            "url": "https://www.youtube.com/watch?v=1MU4wjcxoR4",
        },
        {
            "title": "Dil Chori – Sonu Ke Titu Ki Sweety",
            "url": "https://www.youtube.com/watch?v=MU9oULiwmaU",
        },
        {
            "title": "Koi Kahe Kehta Rahe – Dil Chahta Hai",
            "url": "https://www.youtube.com/watch?v=4vEBkbkzwR8",
        },
    ],
}


def load_css():
    st.markdown(
        """
    <style>
    * {
        font-family: 'Segoe UI', Arial, sans-serif !important;
    }
    
    .main-header {
        font-size: 2.8rem;
        color: #ff5722;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(45deg, #ff5722, #FF9800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        // animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    
    .hindi-title {
        font-size: 1.5rem;
        color: #333;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .subheader {
        font-size: 1.8rem;
        color: #ff5722;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #ff5722;
        padding-bottom: 0.5rem;
    }
    
    .emotion-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        background-color: #2596be;
        border-left: 5px solid #ff5722;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .emotion-card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }
    
    .recommendation-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        background-color: #009933;
        border-left: 6px solid #ff5722;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s;
    }
    
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    .stButton>button {
        background-color: #ff5722;
        color: white;
        border-radius: 20px;
        font-size: 1rem;
        padding: 0.5rem 1.5rem;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .like-button {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .mood-button {
        text-align: center;
        margin-bottom: 12px;
    }
    
    .mood-emoji {
        font-size: 2.2rem;
        margin-bottom: 8px;
        display: block;
    }
    
    .mood-label {
        display: block;
        text-align: center;
    }
    
    .stSelectbox>div>div {
        background-color: #06c91a;
        border: 1px solid #ddd;
        border-radius: 8px;
    }
    
    .info-box {
        background-color: #009933;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .feature-card {
        background-color: #3399ff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        border-top: 4px solid #ff5722;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: #ff5722;
    }
    
    .sidebar-info {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .sidebar-header {
        color: #ff5722;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .step-container {
        display: flex;
        margin-bottom: 1rem;
    }
    
    .step-number {
        background-color: #ff5722;
        color: white;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        margin-right: 10px;
        flex-shrink: 0;
    }
    
    .step-content {
        flex: 1;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        font-size: 0.9rem;
        color: #777;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated-fade {
        animation: fadeIn 0.5s ease-out;
    }
    
    .stVideo {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .hide-on-secondary-pages {
        display: none;
    }
    
    .show-on-home {
        display: block;
    }

    .custom-mood-btn {
        width: 100%; 
        border-radius: 12px;
        padding: 10px;
        background-color: #ffffff;
        color: #333;
        border: 1px solid #ddd;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }
    
    .custom-mood-btn:hover {
        background-color: #ffffff;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def load_user_preferences():
    if not os.path.exists(USER_PREF_PATH):
        return {"liked_songs": [], "emotion_history": []}

    try:
        with open(USER_PREF_PATH, "r") as f:
            return json.load(f)
    except:
        return {"liked_songs": [], "emotion_history": []}


@lru_cache(maxsize=4)
def load_face_model(model_name):
    return model_name


def preprocess_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    brightened = cv2.convertScaleAbs(img_array, alpha=1.5, beta=30)

    return {"original": img_array, "enhanced": enhanced, "brightened": brightened}


def detect_faces_with_fallback(img_array):
    faces = None
    detection_method = "none"

    preprocessed = preprocess_image(img_array)

    try:
        faces = RetinaFace.detect_faces(preprocessed["original"])
        if faces:
            detection_method = "retinaface_original"
            return faces, detection_method
    except Exception:
        pass

    try:
        faces = RetinaFace.detect_faces(preprocessed["enhanced"])
        if faces:
            detection_method = "retinaface_enhanced"
            return faces, detection_method
    except Exception:
        pass

    try:
        faces = RetinaFace.detect_faces(preprocessed["brightened"])
        if faces:
            detection_method = "retinaface_brightened"
            return faces, detection_method
    except Exception:
        pass

    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        opencv_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(opencv_faces) > 0:
            faces = {}
            for i, (x, y, w, h) in enumerate(opencv_faces):
                faces[f"face_{i+1}"] = {"facial_area": (x, y, w, h), "score": 0.9}
            detection_method = "opencv_haar"
            return faces, detection_method
    except Exception:
        pass

    try:
        result = DeepFace.extract_faces(
            img_path=img_array,
            enforce_detection=False,
            detector_backend="opencv",
            align=True,
        )

        if result and len(result) > 0:
            faces = {}
            for i, face_data in enumerate(result):
                facial_area = face_data.get("facial_area", {})
                x = facial_area.get("x", 0)
                y = facial_area.get("y", 0)
                w = facial_area.get("w", 0)
                h = facial_area.get("h", 0)
                faces[f"face_{i+1}"] = {"facial_area": (x, y, w, h), "score": 0.8}
            detection_method = "deepface_detector"
            return faces, detection_method
    except Exception:
        pass

    if not faces:
        try:
            from deepface.detectors import MTCNN

            detector = MTCNN()
            dets = detector.detect_faces(img_array)
            if len(dets) > 0:
                faces = {}
                detection_method = "mtcnn_fallback"
                for i, d in enumerate(dets):
                    x, y, w, h = d["box"]
                    faces[f"face_{i+1}"] = {
                        "facial_area": (x, y, w, h),
                        "score": d.get("confidence", 0.8),
                    }
                return faces, detection_method
        except Exception:
            pass

    return None, detection_method


def analyze_with_models(img_array):
    results = []
    models = ["VGG-Face", "Facenet", "OpenFace", "DeepID", "ArcFace", "Dlib"]
    for model in models:
        try:
            result = DeepFace.analyze(
                img_path=img_array,
                actions=("age", "gender", "race", "emotion"),
                enforce_detection=False,  
                detector_backend="retinaface",
            )
            if isinstance(result, list):
                result = result[0]
            results.append(result)
        except Exception as e:
            st.error(f"Error Analyzing With Model {model}: {str(e)}")
    return results


def weighted_average_results(results):
    if not results:
        return None
        
    avg_result = results[0].copy()
    weights = [1.0] * len(results)

    for i, res in enumerate(results[1:], start=1):
        if "age" in res and "age" in avg_result:
            avg_result["age"] += res["age"] * weights[i]
        if "gender" in res and "gender" in avg_result:
            if res["gender"] == "Woman":
                avg_result["gender"] = "Woman"
        if "race" in res and "race" in avg_result:
            for key in avg_result["race"]:
                if key in res["race"]:
                    avg_result["race"][key] += res["race"][key] * weights[i]
        if "emotion" in res and "emotion" in avg_result:
            for key in avg_result["emotion"]:
                if key in res["emotion"]:
                    avg_result["emotion"][key] += res["emotion"][key] * weights[i]

    total_weight = sum(weights)
    if "age" in avg_result:
        avg_result["age"] /= total_weight
    if "race" in avg_result:
        for key in avg_result["race"]:
            avg_result["race"][key] /= total_weight
    if "emotion" in avg_result:
        for key in avg_result["emotion"]:
            avg_result["emotion"][key] /= total_weight
        avg_result["dominant_emotion"] = max(avg_result["emotion"], key=avg_result["emotion"].get)

    return avg_result


def analyze_emotion_with_models(img_array):
    faces, detection_method = detect_faces_with_fallback(img_array)

    if not faces:
        return (
            None,
            0,
            "No face was detected. Please use better lighting or adjust your face position.",
        )

    try:
        if isinstance(faces, dict) and len(faces) > 0:
            face_key = list(faces.keys())[0]
            face = faces[face_key]
            x, y, w, h = face["facial_area"]

            x, y = max(0, x), max(0, y)
            w = min(w, img_array.shape[1] - x)
            h = min(h, img_array.shape[0] - y)

            if w > 0 and h > 0:
                face_img = img_array[y : y + h, x : x + w]
            else:
                face_img = img_array
        else:
            face_img = img_array
            
        results = analyze_with_models(face_img)
        
        if not results:
            return (
                None,
                0,
                "A face was detected, but we couldn't analyze the emotion. Please try a different angle or change your expression.",
            )
            
        result_dict = weighted_average_results(results)
        if not result_dict or "emotion" not in result_dict:
            return (
                None,
                0,
                "We found your face, but couldn't process the emotion data. Please try again with better lighting.",
            )
            
        dominant_emotion = result_dict["dominant_emotion"]
        confidence = result_dict["emotion"][dominant_emotion] * 100
        
        if confidence < 35:
            return (
                None,
                confidence,
                "We found your face, but the confidence was too low to determine your mood. Please try a different angle or select your mood manually."
            )
            
        emotion_results = {"emotions": result_dict["emotion"], "dominant_emotion": dominant_emotion}
        return emotion_results, confidence, detection_method
            
    except Exception as e:
        st.error(f"Error in emotion analysis: {str(e)}")
        return (
            None,
            0,
            "An error occurred during emotion analysis. Please try again."
        )


def music_recommendation():
    st.markdown(
        '<p class="subheader">Music that matches your mood</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="info-box">📝 <b>How it works:</b> Choose your mood below or take a photo so we can see how you feel. We will suggest songs that match your mood.</div>',
        unsafe_allow_html=True,
    )

    user_prefs = load_user_preferences()

    st.markdown("### Select Your Mood:")
    st.markdown('<div class="animated-fade">', unsafe_allow_html=True)
    st.write("Click on how you feel right now:")

    all_moods = {
        "happy": {"emoji": "😊", "label": "Happy"},
        "sad": {"emoji": "😢", "label": "Sad"},
        "angry": {"emoji": "😠", "label": "Angry"},
        "fear": {"emoji": "😨", "label": "Fear"},
        "neutral": {"emoji": "😐", "label": "Neutral"},
        "disgust": {"emoji": "🤢", "label": "Disgust"},
        "surprise": {"emoji": "😲", "label": "Surprise"},
    }

    mood_selected = None

    cols_row1 = st.columns(4)
    mood_keys = list(all_moods.keys())

    for i in range(4):
        mood = mood_keys[i]
        mood_info = all_moods[mood]
        with cols_row1[i]:
            st.markdown('<div class="mood-button">', unsafe_allow_html=True)
            if st.button(
                f"{mood_info['emoji']} {mood_info['label']}", key=f"mood_{mood}"
            ):
                mood_selected = mood
            st.markdown("</div>", unsafe_allow_html=True)

    cols_row2 = st.columns(3)
    for i in range(4, len(mood_keys)):
        col_idx = i - 4
        mood = mood_keys[i]
        mood_info = all_moods[mood]
        with cols_row2[col_idx]:
            st.markdown('<div class="mood-button">', unsafe_allow_html=True)
            if st.button(
                f"{mood_info['emoji']} {mood_info['label']}", key=f"mood_{mood}"
            ):
                mood_selected = mood
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 2rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

    st.markdown('<div class="animated-fade">', unsafe_allow_html=True)
    st.markdown("### Or Take or Upload a Photo")
    st.write("Let our AI check your face to see how you're feeling.")

    col1, col2 = st.columns([1, 1])

    with col1:
        use_camera = st.checkbox("Use Camera", value=False)
        img_file = None
        if use_camera:
            img_file = st.camera_input("Take a photo", key="camera_input")

        st.write("OR")
        uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

        with st.expander("📷 Tips to help detect faces better:"):
            st.markdown(
                """
            - Make sure your face is well-lit
            - Look directly at the camera
            - Remove sunglasses or hats
            - Keep your face in the center of the frame
            - If your face isn't detected, try a different angle
            """
            )
    st.markdown("</div>", unsafe_allow_html=True)

    if mood_selected:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        user_prefs["emotion_history"].append(
            {
                "emotion": mood_selected,
                "confidence": 100.0,
                "timestamp": timestamp,
                "selection": "manual",
            }
        )

        save_user_preferences(user_prefs)

        st.markdown('<div class="animated-fade">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="emotion-card"><h3>{all_moods[mood_selected]["emoji"]} Your Selected Mood</h3>'
            f"<p>You're feeling: <strong>{mood_selected.capitalize()}</strong></p></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        show_hindi_recommendations(mood_selected, user_prefs)

    image_source = img_file or uploaded_file

    if image_source is not None:
        with st.spinner("Our AI is checking your mood..."):
            img = Image.open(image_source)
            img_array = np.array(img)

            progress_bar = st.progress(0)
            progress_bar.progress(25)

            emotion_results, confidence, detection_method = analyze_emotion_with_models(
                img_array
            )
            progress_bar.progress(75)

            with col2:
                st.image(img, caption="Your photo", use_container_width=True)

            if emotion_results:
                if confidence > 25:
                    dominant_emotion = emotion_results["dominant_emotion"]

                    emotion_emojis = {
                        "happy": "😊",
                        "sad": "😢",
                        "angry": "😠",
                        "fear": "😨",
                        "neutral": "😐",
                        "disgust": "🤢",
                        "surprise": "😲",
                    }
                    emoji = emotion_emojis.get(dominant_emotion, "")

                    st.markdown('<div class="animated-fade">', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="emotion-card"><h3>{emoji} Your mood check</h3>'
                        f"<p>We detected that you're feeling: <strong>{dominant_emotion.capitalize()}</strong> "
                        f"(Confidence: {confidence:.1f}%)</p></div>",
                        unsafe_allow_html=True,
                    )

                    emotions_df = pd.DataFrame(
                        {
                            "Emotion": list(emotion_results["emotions"].keys()),
                            "Confidence": list(emotion_results["emotions"].values()),
                        }
                    )

                    fig = px.bar(
                        emotions_df,
                        x="Emotion",
                        y="Confidence",
                        color="Emotion",
                        title="Your emotion results.",
                        labels={
                            "Confidence": "Confidence %",
                            "Emotion": "Emotions Detected",
                        },
                    )
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Segoe UI, Arial, sans-serif"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    progress_bar.progress(100)

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    user_prefs["emotion_history"].append(
                        {
                            "emotion": dominant_emotion,
                            "confidence": confidence,
                            "timestamp": timestamp,
                            "selection": "ai",
                        }
                    )

                    if len(user_prefs["emotion_history"]) > 50:
                        user_prefs["emotion_history"] = user_prefs["emotion_history"][
                            -50:
                        ]

                    save_user_preferences(user_prefs)

                    show_hindi_recommendations(dominant_emotion, user_prefs)
                else:
                    st.warning(
                        f"We found your face but the confidence ({confidence:.1f}%) "
                        "is too low to determine your mood. Please try again or select manually."
                    )
            else:
                if detection_method != "none":
                    st.warning(
                        "We found your face, but we couldn't clearly determine your mood."
                    )
                else:
                    st.warning(
                        "We didn't detect a face in your picture. Please try again."
                    )

                st.markdown('<div class="animated-fade">', unsafe_allow_html=True)
                st.markdown("### Let's choose your mood manually instead.")
                st.write(
                    "Don't worry! You can still get great song recommendations by telling us how you feel:"
                )

                available_emotions = list(hindi_music_recommendations.keys())
                selected_emotion = st.selectbox(
                    "How are you feeling right now?",
                    available_emotions,
                    format_func=lambda x: x.capitalize(),
                )

                if st.button("Get Music Recommendations", key="manual_mood"):
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    user_prefs["emotion_history"].append(
                        {
                            "emotion": selected_emotion,
                            "confidence": 100.0,
                            "timestamp": timestamp,
                            "selection": "manual",
                        }
                    )

                    if len(user_prefs["emotion_history"]) > 50:
                        user_prefs["emotion_history"] = user_prefs["emotion_history"][
                            -50:
                        ]

                    save_user_preferences(user_prefs)

                    show_hindi_recommendations(selected_emotion, user_prefs)
                st.markdown("</div>", unsafe_allow_html=True)


def show_hindi_recommendations(emotion, user_prefs):
    emotion_emojis = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "fear": "😨",
        "neutral": "😐",
        "disgust": "🤢",
        "surprise": "😲",
    }
    emoji = emotion_emojis.get(emotion, "")

    st.markdown('<div class="animated-fade">', unsafe_allow_html=True)
    st.markdown(
        f"<h3>{emoji} Songs for your {emotion.capitalize()} Mood</h3>",
        unsafe_allow_html=True,
    )
    st.write(
        "Here are songs that match your current mood. Click the heart to save your favorites."
    )

    recommendations = get_personalized_hindi_recommendations(emotion, user_prefs)

    if not recommendations:
        st.warning("No songs were found for this mood. Please choose another one.")
        return

    for i, song in enumerate(recommendations[:3]):
        with st.container():
            st.markdown(
                f"""
            <div class="recommendation-card">
                <h4 class="hindi-title">{song['title']}</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            try:
                st.video(song["url"])
            except Exception:
                st.error(f"Sorry, we couldn't load this video: {song['title']}")
                continue

            col1, col2 = st.columns([1, 6])
            with col1:
                st.markdown('<div class="like-button">', unsafe_allow_html=True)
                if st.button("😍", key=f"like_{emotion}_{i}"):
                    if song not in user_prefs["liked_songs"]:
                        user_prefs["liked_songs"].append(song)
                        st.success("Added to your favorites!")
                        save_user_preferences(user_prefs)
                st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄 Find Different Songs", key=f"refresh_{emotion}"):
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='footer'>Songs chosen based on your mood and what you like.</div>",
        unsafe_allow_html=True,
    )


def get_personalized_hindi_recommendations(emotion, user_prefs):
    if emotion not in hindi_music_recommendations:
        return []

    all_recommendations = hindi_music_recommendations[emotion].copy()

    if not all_recommendations:
        return []

    liked_urls = [song["url"] for song in user_prefs["liked_songs"]]
    new_recommendations = [
        song for song in all_recommendations if song["url"] not in liked_urls
    ]
    previously_liked = [
        song for song in all_recommendations if song["url"] in liked_urls
    ]

    personalized = new_recommendations + previously_liked

    if len(personalized) < 3:
        other_emotions = [e for e in hindi_music_recommendations.keys() if e != emotion]
        for other_emotion in other_emotions:
            other_songs = [
                song
                for song in hindi_music_recommendations[other_emotion]
                if song["url"] not in liked_urls
            ]
            personalized.extend(other_songs)
            if len(personalized) >= 5:
                break

    if len(new_recommendations) > 3:
        random.shuffle(new_recommendations)
        result = new_recommendations[:3]
    else:
        result = new_recommendations[:]
        if previously_liked and len(result) < 3:
            random.shuffle(previously_liked)
            result.extend(previously_liked[: 3 - len(result)])

    return result


def show_mood_history():
    st.markdown('<p class="subheader">Your Mood History</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">📊 <b>Your Personal Mood Tracker:</b> Watch how your feelings change over time. This helps us pick better songs for you.</div>',
        unsafe_allow_html=True,
    )

    user_prefs = load_user_preferences()
    history = user_prefs.get("emotion_history", [])

    if not history:
        st.info(
            "You don't have any mood history yet. Try the Music Recommendation feature to start building your profile."
        )
        return

    df = pd.DataFrame(history)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["source"] = df.apply(
        lambda row: "Manual Selection"
        if row.get("selection") == "manual"
        else "AI Detection",
        axis=1,
    )

    st.markdown('<div class="animated-fade">', unsafe_allow_html=True)
    st.subheader("Your Mood Over Time")

    fig = px.line(
        df,
        x="timestamp",
        y="confidence",
        color="emotion",
        symbol="source",
        title="Your Mood History",
        labels={
            "timestamp": "Time",
            "confidence": "Confidence %",
            "emotion": "Emotion",
        },
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Arial, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    emotion_counts = df["emotion"].value_counts().reset_index()
    emotion_counts.columns = ["Emotion", "Count"]

    fig2 = px.pie(
        emotion_counts, values="Count", names="Emotion", title="Your Most Common Moods"
    )
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Arial, sans-serif"),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="animated-fade">', unsafe_allow_html=True)
    st.subheader("Your Recent Mood Detections")
    recent_df = df.sort_values("timestamp", ascending=False).head(10)
    recent_df["Date"] = recent_df["timestamp"].dt.strftime("%Y-%m-%d")
    recent_df["Time"] = recent_df["timestamp"].dt.strftime("%H:%M:%S")

    recent_df = recent_df.rename(
        columns={
            "emotion": "Mood",
            "confidence": "Confidence %",
            "source": "Detection Method",
        }
    )

    st.dataframe(
        recent_df[["Date", "Time", "Mood", "Confidence %", "Detection Method"]],
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main(): 
    load_css()
    
    if GPU_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div class="sidebar-header">🔥 GPU Acceleration Active</div>', unsafe_allow_html=True)
        st.sidebar.info(f"Using GPU for faster processing:\n{GPU_INFO}")
    
    menu = ["Home", "Music Recommendation", "Mood History"]

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header" style="text-align: center;">Menu</div>', unsafe_allow_html=True
        )
        choice = st.selectbox("Choose a Page", menu)

        st.markdown("---")
        st.markdown(
            '<div class="sidebar-header">How It Works</div>', unsafe_allow_html=True
        )

        st.markdown(
            """
        <div class="step-container">
            <div class="step-number">1</div>
            <div class="step-content">Take a photo or select your mood</div>
        </div>
        <div class="step-container">
            <div class="step-number">2</div>
            <div class="step-content">Our AI analyzes your feelings</div>
        </div>
        <div class="step-container">
            <div class="step-number">3</div>
            <div class="step-content">Get songs that match your mood</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown('<div class="sidebar-header">About</div>', unsafe_allow_html=True)
        st.write(
            "We use AI to understand your emotions and suggest music that fits your mood."
        )

    if choice == "Home":
        st.markdown(
            '<h1 class="main-header">Music Recommendation System</h1>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="info-box">
        Welcome to your personal music recommendation system! This app uses AI to detect your mood and suggest songs that match how you feel right now.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="animated-fade">', unsafe_allow_html=True)
        st.markdown('<p class="subheader">Features</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3>Mood Detection</h3>
                <p>Our AI detects your emotions from a single photo.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
            <div class="feature-card">
                <div class="feature-icon">🎼</div>
                <h3>Music Recommendations</h3>
                <p>Get song recommendations based on your mood.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div class="feature-card">
                <div class="feature-icon">😍</div>
                <h3>Save Favorites</h3>
                <p>Choose songs you love to create your collection.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Track Your Moods</h3>
                <p>Watch your feelings change over time with charts.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("### Available Moods")
        mood_col1, mood_col2 = st.columns(2)

        with mood_col1:
            st.markdown(
                """
            - 😊 **Happy** - Cheerful songs to boost your mood.
            - 😢 **Sad** - Emotional songs for your deep moments.
            - 😠 **Angry** - Strong tracks to match your energy.
            """
            )

        with mood_col2:
            st.markdown(
                """
            - 😐 **Neutral** - Calming songs for everyday listening.
            - 😲 **Surprise** - Tracks to spark your curiosity.
            - 🤢 **Disgust** - Cheerful songs to lift your mood.
            """
            )
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🎼 Get Music Recommendations"):
                st.session_state.choice = "Music Recommendation"
                st.rerun()
        with col2:
            if st.button("📊 View Your Mood History"):
                st.session_state.choice = "Mood History"
                st.rerun()

    elif choice == "Music Recommendation":
        music_recommendation()
    elif choice == "Mood History":
        show_mood_history()


if __name__ == "__main__":
    main()

