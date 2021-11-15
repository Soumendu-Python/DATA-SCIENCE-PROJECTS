# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:11:17 2021

@author: soumendu
"""
import tkinter as tk
from tkinter import messagebox
import cv2
import matplotlib.pyplot as plt
import operator
import numpy as np
import os
os.chdir(r'C:\Post Graduate Course in Data Analytics\SIGN LANGUAGE GESTURE RECOGNITION')
from keras.models import load_model

classifier=load_model('sign_language_gesture_recognition.h5')
classifier.load_weights('sign_language_categorical_weights.h5')

window=tk.Tk()
window.geometry("240x100")
window.title('Sign Language Gesture Recognition Login')
window.resizable(0,0)
#window.config(background='blue')

window.columnconfigure(0,weight=1)
window.columnconfigure(1,weight=3)

username_label=tk.Label(window,text='Username')
username_label.grid(column=0,row=0,sticky=tk.W,padx=5,pady=5)
username_entry=tk.Entry(window)
username_entry.grid(column=1,row=0,sticky=tk.E,padx=5,pady=5)

password_label=tk.Label(window,text='Password')
password_label.grid(column=0,row=1,sticky=tk.W,padx=5,pady=5)
password_entry=tk.Entry(window,show='*')
password_entry.grid(column=1,row=1,sticky=tk.E,padx=5,pady=5)

def slgr():
    if (username_entry.get()=='Soumendu' and password_entry.get()=='somu@2310'):
        messagebox.showinfo('Result','Welcome '+str(username_entry.get()))
        cap=cv2.VideoCapture(0)

        while True:
            ret,frame=cap.read()
            frame=cv2.flip(frame,1)

            roi=frame[120:400,320:620]
            roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            roi=cv2.GaussianBlur(roi,(5,5),2)
            roi=cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            _,roi=cv2.threshold(roi,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            cv2.imshow('roi',roi)

            roi=cv2.resize(roi,(64,64))
            copy=frame.copy()
            cv2.rectangle(copy,(320,120),(620,400),(255,0,0),5)

            result=classifier.predict(roi.reshape(1,64,64,1))
            prediction={'DONE':result[0][0],'HELLO':result[0][1],
                        'LEFT':result[0][2],'NO':result[0][3],
                        '':result[0][4],'RIGHT':result[0][5],
                        'THANK YOU':result[0][6],'YES':result[0][7]}
            predicted=sorted(prediction.items(),key=operator.itemgetter(1),reverse=True)
            res=predicted[0][0]
            cv2.rectangle(copy,(25,45),(590,115),(255,255,255),-1)
            cv2.putText(copy,'Please Put Up Your Right Hand',(30,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            cv2.putText(copy,'In The Box',(30,110),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            cv2.putText(copy,"User Response: "+res,(10,420),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            cv2.imshow('Sign Language Gesture Recognition',copy)

            if cv2.waitKey(1)==13:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        messagebox.showinfo('Error','Authorized Personnel Only')

login_button=tk.Button(window,text='Login',command=slgr)
login_button.grid(column=1,row=3,sticky=tk.E,padx=5,pady=5)

exit_button=tk.Button(window,text='Exit',command=window.destroy)
exit_button.grid(column=2,row=3,sticky=tk.W,padx=5,pady=5)

window.mainloop()