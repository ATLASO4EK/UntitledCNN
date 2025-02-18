from tkinter import *
from tkinter import ttk
from API import *

root = Tk()
root.geometry('400x400')
root.title('FaceIdentifier')
root.resizable(False, False)

cl_model, lc_model = load_model_state_dicts('','')
def get_response():
    cl_pr, face_img = get_face_class('', cl_model, lc_model)
    label['text'] = cl_pr
    face_img.show()

label = ttk.Label()
entry = ttk.Entry()
btn = ttk.Button(text='Get Response', command=get_response)

entry.pack()
btn.pack()
label.pack()




root.mainloop()