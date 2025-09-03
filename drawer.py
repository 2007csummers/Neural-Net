#Create a GUI used to make hand drawn digits


#ABCDEFGHIJKLMNOPQRSTUVWXYZ

from tkinter import *
from PIL import ImageGrab
import bmpHandler
import Main
import tensorflow as tf

#############Functions and Variables##############

prevPoint = [0, 0]
currentPoint = [0, 0]


penColor = "white"
stroke = 30

canvas_data = []

prediction = "N/A"

model = tf.keras.models.load_model("mnist_model.keras")
prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


def clear():
    canvas.delete("all")
    feedback.config(text=f"prediction: N/A")
    return

def paint(event):
    global prevPoint
    global currentPoint

    x = event.x
    y = event.y

    currentPoint = [x, y]

    x0 = x - stroke
    x1 = x + stroke
    y0 = y - stroke
    y1 = y + stroke

    if prevPoint != [0, 0]:
        canvas.create_oval(
            x0,
            y0,
            x1,
            y1,
            fill=penColor,
            outline=penColor,
            width=stroke,
        )

    prevPoint = currentPoint

    if event.type == "5":
        prevPoint = [0, 0]

    return

def submission(Canvas, feedback_label):
    global prediction

    canvas_x, canvas_y = Canvas.winfo_rootx(), Canvas.winfo_rooty()

    image_location = "digit_image.bmp"
    end_image_location = "truncated.bmp" 

    #screen shots the canvas in the drawer window application
    ImageGrab.grab((canvas_x, canvas_y, canvas_x + 900, canvas_y + 900)).save(image_location)

    #truncates the grabbed image down to a 28x28 bmp file and stores at new location
    bmpHandler.truncate(image_location, end_image_location)

    #runs the model on the truncated file
    trunc_arr = bmpHandler.imgtoarr("truncated.bmp")

    prediction = Main.predict(trunc_arr, prob_model)
    print(prediction)

    feedback_label.config(text=f"prediction: {prediction}")
    
    return 


#Create all window objects needed for project. first listed will appear above others.
instruct = Tk()
root = Tk()
#Specify Window
root.title("28x28 Hand Drawn Digits Interface")
root.minsize(1000, 1100) #Width, height
root.geometry("1000x1100")
root.resizable(False, False)


#############Create Frames##############

#Header Frame
frame1 = Frame(root, height=50, width=1000)
frame1.grid(row=0, column=0)

# Holder Frame
holder = Frame(frame1, height=50, width=1000, bg="white", padx=6, pady=10)
holder.grid(row=0, column=0, sticky=NW)
holder.place(relx=0.5, rely=0.5, anchor=CENTER)

frame2 = Frame(root, height=900, width=1000)
frame2.grid(row=1, column=0)

frame3 = Frame(root, height=150, width=1000, bg="white", padx=6, pady=10)
frame3.grid(row=2, column=0)


##############Propogate the Header With Necessary Features###############
header_text = Label(holder, text="Draw Your Digit", borderwidth=1, relief=SOLID, width=30)
header_text.grid(row=0, column=0)

reset_canvas = Button(holder, text="Clear", height=1, width=20, command=clear)
reset_canvas.grid(row=1, column=0)


##############Make Canvas#################
canvas = Canvas(frame2, height=900, width=900, bg="black")
canvas.grid(row=0, column=0)
canvas.place(relx=.5, rely=.5, anchor=CENTER)
canvas.config(cursor="pencil")

#event Binding
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", paint)
canvas.bind("<Button-1>", paint)


#############Create Submission Button and Feedback label ###############
feedback = Label(frame3, text=f"Prediction: {prediction}!")
feedback.grid(row=1, column=0)

submit = Button(frame3, text="Submit", height=1, width=6, command= lambda: submission(canvas, feedback))
submit.grid(row=0, column=0)


################### Create Second Window for instructions ####################

#Specify instruction window
instruct.title("User Manual")
instruct.minsize(700, 200)
instruct.geometry("700x200")
instruct.resizable(False, False)

IFrame1 = Frame(instruct, height=200, width=700, padx=50)
IFrame1.grid(row=0, column=0)


instruct_text = "Welcome to the Number Reader.\n\nTo start, Draw a single digit number on the black portion of the window. Try to keep the number centered, and don't draw too big or too small.\n\nClick the submit button to have your number read. Result will appear below the submit button.\n\nClick the clear button to clear the canvas and the prediction"

instructions = Label(IFrame1, text=instruct_text, justify=CENTER, wraplength=600)
instructions.place(relx=.5, rely=.5, anchor=CENTER)
instructions.pack(pady=10, padx=10)
instructions.grid(row=0, column=1)




root.mainloop()
instructions.mainloop()


