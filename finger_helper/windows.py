from PIL import Image, ImageTk, ImageOps, ImageFont, ImageDraw
import tkinter as tk
from tkinter import ttk
import cv2
import os
import logging
from tkinter import filedialog, colorchooser
import threading
import subprocess
import platform
from fastai.vision.all import *
import csv
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import os

#TODO make folder so that pics, model, results etc. are all contained in their own folder

class ImageCaptureWindow(tk.Toplevel):
    def __init__(self, parent):
        """ Initialize frame which uses OpenCV + Tkinter. 
            The frame:
            - Uses OpenCV video capture and periodically captures an image
              and show it in Tkinter
            - A save button allows saving the current image to output_path.
            - Consecutive numbering is used, starting at 00000001.jpg.
            - Checks if output_path exists, creates folder if not.
            - Checks if images are already present and continues numbering.
            
            attributes:
                vs (cv2 VideoSource): webcam to capture images from
                output_path (str): folder to save images to.
                current_image (PIL Image): current image displayed
                save_btn (ttk Button): press to save image
                burst_btn (ttk Button): press for burst mode
                browse_btn (ttk Button): press to change directory
                panel (ttk Label): to display image in frame
        """
        super().__init__(parent)
        self.grid()

        # Bool for freezing camera feed after picture is taken
        self.freeze = False
        
        # 0 is your default video camera
        self.vs = cv2.VideoCapture(0) 

        
        # Prepare an attribute for the image
        self.current_image = None 

         # Button to save current image to file
        save_btn = ttk.Button(self, text="Save", command=self.save_image)
        save_btn.grid(row=0, column=0, sticky='nw', padx=20, pady=10)

        # Change directory button
        browse_btn = ttk.Button(self, text="Browse", command=self.set_dir_btn)
        browse_btn.grid(row=0, column=0, sticky='ne', padx=20, pady=10)

        # Burst button
        burst_btn = ttk.Button(self, text="Burst", command=self.thread_caller)
        burst_btn.grid(row=0, column=0, sticky='n', padx=20, pady=10)

        # Currect directory label
        self.relpath = tk.StringVar()
        directory_lab = ttk.Label(self, textvariable=self.relpath)
        directory_lab.grid(row=1, column=0, pady=0)
        self.set_dir_btn()
        
        # Label to display image
        self.panel = ttk.Label(self)  
        self.panel.grid(row=2, column=0,padx=10, pady=10)

        # start the display image loop
        self.video_loop()

        
    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter 
            
            The image is processed using PIL: 
            - crop left and right to make image smaller
            - mirror 
            - convert to Tkinter image
            
            Uses after() to call itself again after 30 msec.
        
        """
        # read frame from video stream
        ok, frame = self.vs.read()  
        # frame captured without any errors
        if ok:  
            # convert colors from BGR (opencv) to RGB (PIL)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            # convert image for PIL
            self.current_image = Image.fromarray(cv2image)  
            # Optional if camera is wide: crop 200 from left and right
            #self.current_image = ImageOps.crop(self.current_image, (200,0,200,0)) 
            # mirror, easier to locate objects
            self.current_image = ImageOps.mirror(self.current_image) 
            # convert image for tkinter for display, scale to 50% of size of original image
            imgtk = ImageTk.PhotoImage(image=ImageOps.scale(self.current_image, 0.5)) 
            # anchor imgtk so it does not get deleted by garbage-collector
            self.panel.imgtk = imgtk  
             # show the image
            self.panel.config(image=imgtk)

        # Pause drawing image as image is saved
        if self.freeze:
            self.after(250, self.video_loop) 
            self.freeze = False
        else:
            self.after(30, self.video_loop) 

    def set_dir_btn(self):
        """
        Add button for popup gui to change directory
        """
        self.output_path = filedialog.askdirectory(initialdir=os.path.join(os.getcwd(), "Projects"), title="Set output directory") + '/'
        self.relpath.set('\\' + os.path.relpath(self.output_path, os.getcwd()))
        count = len(os.listdir(self.output_path))
        logging.info("Change to {}".format(self.output_path))
        logging.info("Current image count in output folder is {}".format(count))

    def burst(self):
        """
        Saves picture at specified interval
        """
        num_pics = 10
        interval = 500
        for _ in range(num_pics):
            self.save_image()
            self.after(interval)   
    
    def thread_caller(self):
        """
        Calls self.burst as thread..?
        """
        threading.Thread(target=self.burst).start()

    def save_image(self):
        """ Save current image to the file 
        
        self.current_image is saved to output_path using
        consecutive numbering using count 
        zero-filled, eight-number format, e.g. 00000001.jpg.
        
        """
        self.freeze = True
        count = len(os.listdir(self.output_path))

        file_length = len(str(count + 1))
        file_name =  '0'*(8-file_length) + str(count + 1) + ".jpg"

        self.current_image.save(self.output_path + file_name)
        logging.info("Saved {}/{}".format(self.relpath.get(), file_name)) # is there a reason to use logging over print?

    def destructor(self):
        """ Destroy the root object and release all resources """
        self.destroy() # close the Tk window
        logging.info("Closing image capture window")
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # close OpenCV windows

class TrainingWindow(tk.Toplevel):
    def __init__(self, parent):
        """
        Helper for trainer

        Attributes:
            digit_path: path to dataset directory. Must be in the form `digit_path/train` and `digit_path/valid`
            batch_size: batch size when training model
            epochs: number of epoch when training model
            seed: seed used for rng of modules
            lr: learning rate when training model
            output_path: path of .pkl model
            cuda: True if cuda PyTorch is installed
            gpu_accel: True if GPU will be used to train model
            confusion: True if confusion matrix is to be saved
            confusion_path: output path of confusion matrix image
            base_model: base model used for transfer learning i.e. resnet18, resnet50...
        """
        super().__init__(parent)
        
        # Instantiate and initialize tkinter variables
        self.digit_path = tk.StringVar()
        self.digit_path.set("pics/digits_1")
        self.batch_size = tk.IntVar()
        self.batch_size.set(16)
        self.epochs = tk.IntVar()
        self.epochs.set(12)
        self.seed = tk.IntVar()
        self.seed.set(69)
        self.lr = tk.DoubleVar()
        self.lr.set(3e-5)
        self.output_path = tk.StringVar()
        self.output_path.set("models/finger_count.pkl")
        self.cuda = tk.BooleanVar()
        if torch.cuda.is_available(): self.cuda.set(True) 
        else: self.cuda.set(False)
        self.gpu_accel = tk.BooleanVar()
        self.gpu_accel.set(self.cuda.get())
        self.confusion = tk.BooleanVar()
        self.confusion.set(True)
        self.confusion_path = tk.StringVar()
        self.confusion_path.set("models/confusion matrix/img1.png")
        self.model_list = [
        'alexnet',
        'densenet121',
        'densenet161',
        'densenet201',
        'efficientnet_b0',
        'efficientnet_b1',
        'googlenet',
        'inception_v3',
        'mnasnet0_5',
        'mnasnet1_0',
        'mnasnet0_75',
        'mnasnet1_3',
        'mobilenetV2',
        'mobilenet_v3_small',
        'resnet18',
        'resnet34',
        'resnet50',
        'squeezenet1_0',
        'squeezenet1_1',
        'vgg13_bn',
        'vgg16',
        'vgg19_bn',]
        
        self.base_model = self.model_list[0]
        self.output_function = tk.StringVar()

        # Frame for all of the model parameters
        # Contains multiple frames made up of either self.create_widget() or self.create_btn()
        self.param_frame = ttk.LabelFrame(self, text="Model parameters", padding=(20, 20))

        # Create the sub frames
        self.btn_frame = ttk.Frame(self.param_frame)
        self.create_btn(self.btn_frame, self.digit_path, True, (0,0), "Picture directory:")                             # Digit directory

        self.widget_frame = ttk.Frame(self.param_frame)
        self.create_widget(self.widget_frame, self.batch_size, ttk.Entry, "Batch size:", 7, (1,0))                      # Batch size
        self.create_widget(self.widget_frame, self.epochs, ttk.Entry, "Number of epochs:", 7, (1, 3))                   # Number of epochs
        self.create_widget(self.widget_frame, self.seed, ttk.Entry, "Seed:", 7, (2,0))                                  # Seed
        self.create_widget(self.widget_frame, self.lr, ttk.Entry, "Learning rate:", 7, (2,3))                           # Learning rate
        self.create_widget(self.widget_frame, self.base_model, ttk.Combobox, "Base model:", 7, (3,2))                   # Base model
        self.create_widget(self.widget_frame, self.gpu_accel, ttk.Checkbutton, "Use GPU:", 0, (4,0))                    # GPU acceleration
        self.create_widget(self.widget_frame, self.confusion, ttk.Checkbutton, "Generate confusion matrix:", 0, (4,3))  # Create confusion matrix

        self.btn_frame2 = ttk.Frame(self.param_frame)
        self.create_btn(self.btn_frame2, self.output_path, True, (0,0), "Model output:")                                # Saved model output
        self.create_btn(self.btn_frame2, self.confusion_path, True, (1,0), "Confusion output:")                         # Confusion matrix output

        # Pack sub frames into the parameter frame
        self.btn_frame.pack()
        self.widget_frame.pack()
        self.btn_frame2.pack()

        # Place paramater frame
        self.param_frame.grid(row=0, column=0, pady=10)

        # Generate the code
        self.generate_btn = ttk.Button(self, text="Generate", command=self.generate)
        self.generate_btn.grid(row=1, column=0, ipadx=200, ipady=7, pady=(2, 10))

        # Entry to view/copy code
        self.function_entry = ttk.Entry(self, textvariable=self.output_function, width=55)
        self.function_entry.grid(row=2, column=0, pady=5)

        # Button to automatically copy code to clipboard
        self.copy_btn = ttk.Button(self, text="Copy to clipboard", command=self.copy)
        self.copy_btn.grid(row=3, column=0, pady=5)


    def create_btn(self, frame: ttk.Frame, attr, mode, loc, text):
        """
        Create a frame to hold data in the form Label - Entry, Button.
        This is for variables which hold a path since there is a browse button associated with it.
        Things get ugly when shared with the 2 column variables (Label - Entry)
        The buttons are `browse` buttons which go along with changing the path of a directory or file
        Label - Entry - Button
        frame ttk.Frame: The frame which the 3 widgets will be placed in
        attr: tk.StringVar which will be set to the path of directory/file
        mode: True => browse directory
              False => browse file
        loc (int, int): (row, column) location of label within the frame
        text: Label text
        """
        # Create the widgets
        label=ttk.Label(frame, text=text, justify='center', width=16)
        entry=ttk.Entry(frame, textvariable=attr)
        if mode:
            button = ttk.Button(frame, text="Browse", command=lambda: self.dir_path(attr))
        else:
            button = ttk.Button(frame, text="Browse", command=lambda: self.save_path(attr))

        # Place the widgets
        label.grid(row=loc[0], column=loc[1], padx=5, pady=5)
        entry.grid(row=loc[0], column=loc[1]+1, ipadx=40, padx=5, pady=5)
        button.grid(row=loc[0], column=loc[1]+2, padx=5, pady=5)


    def create_widget(self, frame: ttk.Frame, attr, widget_type:ttk.Widget, text, width, loc):
        """
        Create a frame which holds pairs of widgets in the form label - widget
        frame ttk.Frame: the frame which the pair of widgets will be placed
        widget_type ttk.Widget: The widget to be used i.e. Entry, Checkbutton...
        attr: the class attribute variable
        text: label text
        width int: width of widget
        loc (int, int): (row, column) of the label within the frame
        """
        label=ttk.Label(frame, text=text)
        if widget_type==ttk.Entry:
            widget = ttk.Entry(frame, textvariable=attr, justify='center', width=width)
        elif widget_type==ttk.Checkbutton:
            widget = ttk.Checkbutton(frame, variable=attr)
            if attr.get()==False: widget.config(state=tk.DISABLED)
        elif widget_type==ttk.Combobox:
            # Only one Combobox so its hardcoded
            widget = ttk.Combobox(frame, textvariable=attr, width=16)
            widget['values'] = self.model_list
            widget.set(widget['values'][0])

        label.grid(row=loc[0], column=loc[1], sticky='e', padx=(15,0), pady=6)
        widget.grid(row=loc[0], column=loc[1]+1, padx=(0,10), pady=6, ipadx=0, sticky='w')

    def dir_path(self, attr: tk.StringVar):
        """Set a StringVar to the relative path for a directory"""
        path = filedialog.askdirectory(initialdir=os.path.join(os.getcwd(), "Projects"), title="Set directory")
        rel_path = os.path.relpath(path)
        attr.set(rel_path)

    def save_path(self, attr: tk.StringVar):
        """Set a StringVar to the relative path of the file"""
        path = filedialog.asksaveasfilename(initialdir=os.path.join(os.getcwd(), "Projects"), title="Set file path")
        rel_path = os.path.relpath(path)
        attr.set(rel_path)

    def generate(self):
        """Generate code ready to copy and paste into trainer.ipynb"""
        self.output_function.set(f"train_model(digit_folder={self.digit_path.get()}, batch_size={self.batch_size.get()}, epochs={self.epochs.get()}, seed={self.seed.get()}, lr={self.lr.get()}, output={self.output_path.get()}, save_conf={self.confusion.get()}, conf_out={self.confusion_path.get()}, model={self.base_model})")

    def copy(self):
        """Copy generated code to clipboard"""
        if platform.system() == "Windows":
            subprocess.run("clip", text=True, input=self.output_function.get())
        else:
            subprocess.run("pbcopy", text=True, input=self.output_function.get())

    def destructor(self):
        """Destroy TrainingWindow"""
        logging.info("Closing model training window")
        self.destroy()

class ProductionWindow(tk.Toplevel):
    def __init__(self, parent):
        """ Initialize frame which uses OpenCV + Tkinter. 
            The frame:
            - Uses OpenCV video capture and periodically captures an image.
            - Uses a fastai learner to predict the finger count
            - Overlays the label and probability on the image
            - and shows it in Tkinter
            
            attributes:
                vs (cv2 VideoSource): webcam to capture images from
                learn (fastai Learner): CNN to generate prediction.
                current_image (PIL Image): current image displayed
                pil_font (PIL ImageFont): font for text overlay
                panel (ttk Label): to display image in frame
        """
        super().__init__(parent)
        
        # 0 is your default video camera
        self.vs = cv2.VideoCapture(0) 
        
        # Use GPU for predict
        self.GPU = tk.BooleanVar()

        # Check if Cuda PyTorch installed
        if torch.cuda.is_available():
            self.cuda = True
            self.GPU.set(True)
        else:
            logging.warning("PyTorch Cuda not found")
            self.cuda = False
            self.GPU.set(False) 

        self.choose_model()

        self.csv_path = None

        self.current_image = None 
        self.pil_font = ImageFont.truetype("finger_helper/fonts/DejaVuSans.ttf", 40)

        self.actual_digit_value = tk.StringVar()
        self.actual_digit_value.set("one")   

        # Set colour of prob text
        self.colour = 'black'

        # Create widgets
        # Panel shows image
        self.panel = ttk.Label(self, image=self.current_image)  

        # Save screenshot button
        self.save_btn = ttk.Button(self, text="Save", command=self.save_data)
        self.save_btn.config(state=tk.DISABLED) # need to set csv first
        
        # Change model button
        self.model_btn = ttk.Button(self, text="Change model", command=self.choose_model)

        # Change colour button
        self.colour_btn = ttk.Button(self, text="Change colour", command=self.change_colour)

        # Change csv button
        self.change_csv_btn = ttk.Button(self, text="Change CSV", command=self.change_csv)

        # Make entered digit label
        self.digit_label = ttk.Label(self, text="Actual digit:")

        # Actual digit entry
        self.actual_digit = ttk.Entry(self, textvariable=self.actual_digit_value)

        # Use GPU checkbox
        self.gpu_enable = ttk.Checkbutton(self, text="Use GPU", command=self.gpu_accel_toggle, variable= self.GPU)
        if not self.cuda:
            self.gpu_enable.config(state=tk.DISABLED)

        # Analyze data
        self.analyze_btn = ttk.Button(self, text="Analyze production data", command=self.open_analyze)
        self.analyze_btn.config(state=tk.DISABLED) # need to set csv first

        
        # Order Widgets
        self.save_btn.pack(fill='x')
        self.model_btn.pack(fill='x')
        self.colour_btn.pack(fill='x')
        self.gpu_enable.pack(side="top")
        self.change_csv_btn.pack()
        self.digit_label.pack()
        self.actual_digit.pack()
        self.panel.pack()
        self.analyze_btn.pack(pady=10)

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop(colour=self.colour)

        
    def video_loop(self, colour='red'):
        """ Get frame from the video stream and show it in Tkinter 
            
            The image is processed using PIL: 
            - crop left and right to make image smaller
            - mirror 
            - convert to Tkinter image
            
            Uses fastai learner to predict label and probability,
            overlayed as text onto image displayed.
            
            Uses after() to call itself again after 20 msec.
        
        """
        # read frame from video stream
        ok, frame = self.vs.read()  
        # frame captured without any errors
        if ok:  
            # convert colors from BGR (opencv) to RGB (PIL)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            # convert image for PIL
            self.current_image = Image.fromarray(cv2image)  
            # camera is wide: crop 200 from left and right
            # self.current_image = ImageOps.crop(self.current_image, (200,0,200,0)) 
            # mirror, easier to locate objects
            self.current_image = ImageOps.mirror(self.current_image) 
            
            #predict
            if self.learn != None:
                self.pred,pred_idx,probs = self.learn.predict(tensor(self.current_image))
                self.pred_prob = round(probs[pred_idx].item(), 2)
                pred_str = f"{self.pred} ({self.pred_prob})"
            else:
                pred_str = "NO MODEL LOADED"
    
            #add text
            draw = ImageDraw.Draw(self.current_image)
            draw.text((10, 10), pred_str, font=self.pil_font, fill=colour)
            
            # convert image for tkinter for display, scale to 50% of size of original image
            imgtk = ImageTk.PhotoImage(image=ImageOps.scale(self.current_image, 0.5)) 
            # anchor imgtk so it does not get deleted by garbage-collector
            self.panel.imgtk = imgtk  
             # show the image
            self.panel.config(image=imgtk)
        
        # do this again after 20 milliseconds
        self.after(20, self.video_loop, self.colour) 

    def destructor(self):
        """ Destroy the root object and release all resources """
        logging.info("Closing production window")
        self.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

    def save_data(self):
        """Save predicted data to csv file"""
        try:
            csv_file = open(self.csv_path, 'a', newline='')
            csv_writer = csv.writer(csv_file)
        except:
            logging.error(f"Could not load {self.csv_path}")

        data_to_write = [self.actual_digit_value.get().strip().lower(), self.pred, self.pred_prob]
        if data_to_write[0] in ['one', 'two', 'three', 'four', 'five']:
            csv_writer.writerow(data_to_write)
            logging.info(f"Data recorded --- {data_to_write}")
        else:
            logging.warning(f"'{data_to_write[0]}' is not a valid classification")
        csv_file.close()

    def choose_model(self):
        """Change the learner used"""
        self.model_path = filedialog.askopenfilename(filetypes=[("Pickle", ".pkl")], initialdir=os.path.join(os.getcwd(), "Projects"), title="Set model file path")
        try:
            self.learn = load_learner(self.model_path, cpu= not self.GPU.get())
            logging.info(f"learner {self.model_path} loaded")
        except FileNotFoundError:
            try:
                self.learn==None
                logging.warning(f"No model found. Resorting to previous model")

            except AttributeError:
                logging.error(f"No Model found")
                self.learn=None
        except:
            logging.error(f"Model {self.model_path} Could not be loaded. Please try again")
            self.choose_model()

    def change_csv(self):
        """Change the CSV saved to"""
        csv_path = filedialog.askopenfilename(filetypes=[("Comma-seprated value", ".csv .CSV")], initialdir=os.path.dirname(os.path.dirname(self.model_path)), title="Select CSV file")
        if csv_path == '':
            logging.warning(f"No CSV found. Resorting to previous file \"{self.csv_path}\"")
        else:
            self.csv_path = csv_path
            logging.info(f"{self.csv_path} loaded")
            self.save_btn.config(state=tk.NORMAL)
            self.analyze_btn.config(state=tk.NORMAL)

    def change_colour(self):
        """Change the label colour"""
        self.colour = colorchooser.askcolor(title="Choose Label Colour")[1]
        logging.info(f"Colour changed to: {self.colour}")

    def gpu_accel_toggle(self):
        """Toggle on and off using GPU acceleration"""
        logging.info(f"Using GPU: {self.GPU.get()}")
        self.learn = load_learner(self.model_path, cpu=not self.GPU.get())

    def open_analyze(self):
        analyze_win = AnalyzeWindow(self, self.csv_path)
        analyze_win.protocol("WM_DELETE_WINDOW", analyze_win.destructor)


class AnalyzeWindow(tk.Toplevel):
    def __init__(self, parent, csv_path):
        super().__init__(parent)
        logging.info("Opening production analysis window")

        # Reload window button
        self.refresh_btn = tk.Button(self, text="Refresh Data", command=self.refresh)

        # Create panel for confusion matrix image
        self.panel = tk.Label(self, image=None)  # Placeholder for image, will be updated later

        # Create labels for the accuracy, avg prob
        self.accuracy_label = tk.Label(self, text="")
        self.probability_label = tk.Label(self, text="")

        # Pack Widgets
        self.refresh_btn.pack(fill="both", pady=5)
        self.panel.pack(side=tk.TOP)
        self.accuracy_label.pack(pady=7)
        self.probability_label.pack(pady=7)

        # Create a Frame to hold the excel-like array and scrollbar
        self.excel_frame = tk.Frame(self)
        self.excel_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Canvas widget for the excel-like array
        self.excel = tk.Canvas(self.excel_frame)
        self.excel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a vertical scrollbar
        scrollbar = ttk.Scrollbar(self.excel_frame, orient=tk.VERTICAL, command=self.excel.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.excel.configure(yscrollcommand=scrollbar.set)

        # Create Frame which will hold the array of data
        self.data_frame = tk.Frame(self.excel)
        self.excel.create_window((0, 0), window=self.data_frame, anchor=tk.NW)

        self.csv_path = csv_path
        self.load_excel(self.csv_path)

        # Bind mouse wheel event to canvas for scrolling
        self.excel.bind_all("<MouseWheel>", self.on_mousewheel)

    def on_mousewheel(self, event):
        self.excel.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_confusion(self, true_val, pred_val, cmap:str="Blues"):
        title = "Confusion Matrix"
        "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs
        labels = ['one', 'two', 'three', 'four', 'five']
        cm = confusion_matrix(true_val, pred_val)
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=90)
        plt.yticks(tick_marks, labels, rotation=0)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f'{cm[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center",
                     color="white"
                     if cm[i, j] > thresh else "black")

        ax = fig.gca()
        ax.set_ylim(len(labels)-.5,-.5)

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)
        # plt.show()

        # Assign to variable
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        img_temp = Image.open(buffer)
        self.matrix_img = ImageTk.PhotoImage(image=ImageOps.scale(img_temp, 0.7))
        img_temp.close()
        buffer.close()
        plt.close()

        # Update panel with confusion matrix image
        self.panel.config(image=self.matrix_img)
        self.panel.image = self.matrix_img

    def load_excel(self, reader):
        """
        Load the excel file, generate confusion matrix, calculate accuracy
        """
        self.csv_file = open(reader, 'r', newline='')
        reader = csv.reader(self.csv_file)
        data = list(reader)
        headers = data[0]

        # Seperate into true, predicted, and probability
        true_vals, pred_vals, prob = zip(*[(sub[0], sub[1], sub[2]) for sub in data[1:]])

        # Calculate accuracy and average probability
        accuracy = f"Accuracy = {100 * accuracy_score(true_vals, pred_vals):.2f}%"
        avg_prob = f"Average probability = {100 * sum([float(num_str) for num_str in prob]) / len(prob):.2f}%"

        # Update accuracy and probability labels
        self.accuracy_label.config(text=accuracy)
        self.probability_label.config(text=avg_prob)

        # Add headers
        for j, header in enumerate(headers):
            header_label = tk.Label(self.data_frame, text=header, borderwidth=1, relief="solid", width=15,
                                    bg="lightgray")
            header_label.grid(row=0, column=j, sticky="nsew")

        # Add data rows
        for i, row in enumerate(data[1:], start=1):
            for j, value in enumerate(row):
                label = tk.Label(self.data_frame, text=value, borderwidth=1, relief="solid", width=15)
                label.grid(row=i, column=j, sticky="nsew")

        self.create_confusion(true_vals, pred_vals)

        # Update canvas scroll region after all widgets are added
        self.data_frame.update_idletasks()
        self.excel.config(scrollregion=self.excel.bbox(tk.ALL))

        self.csv_file.close()

    def refresh(self):
        self.load_excel(self.csv_path)

    def destructor(self):
        """Destroy AnalyzeWindow"""
        logging.info("Closing production analysis window")
        self.destroy()