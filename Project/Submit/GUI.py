import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import ttk
from PIL import ImageTk, Image
import gui_backend
from skimage import io

class PredictImage():
    def __init__(self, base):
        self.base = base
        self.frame1 = tk.Frame(self.base)
        self.frame1.grid(row=0, column=0)
        self.frame2 = tk.Frame(self.base)
        self.frame2.grid(row=1, column=0)
        self.panel = tk.Label(self.frame2)
        self.result = tk.Label(self.frame2)

        close_btn = tk.Button(self.frame1, text = "Close", command = window.quit) # closing the 'window' when you click the button
        close_btn.grid(row=0, column=1, pady=10)
        open_file_btn = tk.Button(self.frame1, text = 'Open File', command = self.display_image)
        open_file_btn.grid(row=0, column=0, pady=10)

    def predict_image(event):
        result = 'calculated result'
        display = tk.Label(self.frame1, text = result)
        display.grid(row=2)

    def display_image(self):
        self.panel.grid_forget()
        self.result.grid_forget()
        filename = askopenfilename()
        img = ImageTk.PhotoImage(Image.open(filename))
        self.panel = tk.Label(self.frame2, image = img)
        self.panel.image = img
        self.panel.grid(row=0, column=0)
        input_img = io.imread(filename)
        prediction = gui_backend.img_predict(input_img)
        str_predicted = ''
        if len(prediction) == 0:
            str_predicted = 'Not Identified'
        else:
            for i, x in enumerate(prediction):
                str_predicted += x
                if i < len(prediction)-1:
                    str_predicted += ', '
        prediction = 'Predicted result: ' + str_predicted
        # prediction = 'calculated result'
        self.result = tk.Label(self.frame2, text = prediction)
        self.result.grid(row=5, column=0)
        
class ViewTop50():
    def __init__(self, base):
        self.base = base
        self.frame1 = tk.Frame(self.base)
        self.frame1.grid(row=0, column=0)
        self.frame2 = tk.Frame(self.base)
        self.frame2.grid(row=1, column=0)

        self.allclass_info = gui_backend.top50()
        self.classes = list(self.allclass_info.keys())
        self.option1 = tk.StringVar(base)
        self.option1.set(self.classes[0])
        self.option1.trace('w', self.change_dropdown1)
        # self.img_layout = [[]] * 25

        # scroll_bar = tk.Scrollbar(self.frame2)
        # scroll_bar.grid(rowspan=25, column=2)

        # mylist = tk.Listbox(self.frame2, yscrollcommand = scroll_bar.set )
        # for line in range(100):
        #     mylist.insert('end', "This is line number " + str(line))

        # mylist.grid()
        # scroll_bar.config( command = self.frame2.yview )

        # for row in range(25):
        #     for col in range(2):
        #         self.img_layout[row].append(tk.Label(self.frame2))

        # keep the code below
        default_class = self.classes[0]
        self.top_img = self.allclass_info[default_class][0]
        self.top_score = self.allclass_info[default_class][1]
        self.option2 = tk.StringVar(base)
        self.option2.set(self.allclass_info[default_class][0][0])
        self.option2.trace('w', self.change_dropdown2)
        class_label = tk.Label(self.frame1, text='Select the class: ').grid(row=0, column=0)
        select_cat = tk.OptionMenu(self.frame1, self.option1, *self.classes)
        select_cat.grid(row=0, column=1)
        img_label = tk.Label(self.frame1, text='Select the image (Top50): ').grid(row=1, column=0)
        self.select_img = tk.OptionMenu(self.frame1, self.option2, *list(self.allclass_info[default_class][0]))
        self.select_img.grid(row=1, column=1)

        self.panel = tk.Label(self.frame2)
        self.prob = tk.Label(self.frame2)

    def change_dropdown1(self, *args):
        self.selected_class = self.option1.get()
        self.top_img = self.allclass_info[self.selected_class][0]
        self.top_score = self.allclass_info[self.selected_class][1]
        # for row in range(25):
        #     for col in range(2):
        #         self.img_layout[row][col].grid_forget()
        # counter = 0
        # for row in range(25):
        #     for col in range(2):
        #         img_file = './JPEGImages/' + top_img[counter] + '.jpg'
        #         #TODO parse the image path
        #         display_img = ImageTk.PhotoImage(Image.open(img_file))
        #         self.img_layout[row][col] = tk.Label(self.frame2, image=display_img)
        #         self.img_layout[row][col].image = display_img
        #         prob = tk.Label(self.frame2, text='Probability: ' + str(top_score[counter]))
        #         prob.grid(row=row*2+1, column=col)
        #         self.img_layout[row][col].grid(row=row*2, column=col)
        #         #TODO display the label as well
        #         counter += 1
        # get selected 50 img filename
        # display all
        self.select_img = tk.OptionMenu(self.frame1, self.option2, *self.allclass_info[self.selected_class][0])
        self.select_img.grid(row=1, column=1)
    
    def change_dropdown2(self, *args):
        self.panel.grid_forget()
        self.prob.grid_forget()
        position = self.top_img.index(self.option2.get())
        img_file = './JPEGImages/' + self.option2.get() + '.jpg'
        # filename is self.option2.get()
        img = ImageTk.PhotoImage(Image.open(img_file))
        self.panel = tk.Label(self.frame2, image = img)
        self.panel.image = img
        self.panel.grid(row=0, column=0)
        self.prob = tk.Label(self.frame2, text='Probability: ' + str(self.top_score[position]))
        self.prob.grid(row=1, column=0)

window = tk.Tk()

tabControl = ttk.Notebook(window)          # Create Tab Control
tab1 = ttk.Frame(tabControl)            # Create a tab 
tab2 = ttk.Frame(tabControl)
tabControl.add(tab1, text='Predict Image')      # Add the tab
tabControl.add(tab2, text='View Top50 Result')
tabControl.pack(expand=1, fill="both")  # Pack to make visible

# w, h = window.winfo_screenwidth(), window.winfo_screenheight()
w = 700
h = 700
window.geometry("%dx%d+0+0" % (w, h))
window.title('Small Project')
pi = PredictImage(tab1)

# get classes and top50 filenames
vt = ViewTop50(tab2)
window.mainloop()