# Adapted from code found in this thread:
# https://stackoverflow.com/questions/32342935/using-opencv-with-tkinter
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk
import cv2
import os
import logging
from tkinter import filedialog, colorchooser
import threading
from fastai.vision.all import *

class ProductionFrame(tk.Toplevel):
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
        

        # Use CPU for predict
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

        
        self.current_image = None 
        self.pil_font = ImageFont.truetype("fonts/DejaVuSans.ttf", 40)


        self.output_path = "screenshots/"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Set colour of prob text
        self.colour = 'black'

        # Create widgets
        # Panel shows image
        self.panel = ttk.Label(self, image=self.current_image)  

        # Save screenshot button
        self.save_btn = ttk.Button(self, text="Save", command=self.screenshot)
        
        # Change model button
        self.model_btn = ttk.Button(self, text="Change model", command=self.choose_model)

        # Change colour button
        self.colour_btn = ttk.Button(self, text="Change colour", command=self.change_colour)

        # Use GPU checkbox
        self.gpu_enable = ttk.Checkbutton(self, text="Use GPU", command=self.gpu_accel_toggle, variable= self.GPU)
        if not self.cuda:
            self.gpu_enable.config(state=tk.DISABLED)

        
        # Order Widgets
        self.save_btn.pack(fill='x')
        self.model_btn.pack(fill='x')
        self.colour_btn.pack(fill='x')
        self.gpu_enable.pack(side="top")
        self.panel.pack()

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
            # camera is wide: crop 200 from left and right
            # self.current_image = ImageOps.crop(self.current_image, (200,0,200,0)) 
            # mirror, easier to locate objects
            self.current_image = ImageOps.mirror(self.current_image) 
            
            #predict
            if self.learn != None:
                pred,pred_idx,probs = self.learn.predict(tensor(self.current_image))
                pred_str = f"{pred} ({probs[pred_idx].item():.2f})"
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
        logging.info("closing GUI...")
        self.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

    def screenshot(self):
        """Take screenshot and save to screenshot folder"""
        pic_count = len(os.listdir(self.output_path))
        file_name =  '0'*(8-len(str(pic_count+1))) + str(pic_count + 1) + ".jpg"
        self.current_image.save(self.output_path + file_name)
        logging.info("Saved screenshots\{}".format(file_name))

    def choose_model(self):
        """Change the learner used"""
        self.model_path = filedialog.askopenfilename()
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


    def change_colour(self):
        """Change the label colour"""
        self.colour = colorchooser.askcolor(title="Choose Label Colour")[1]
        logging.info(f"Colour changed to: {self.colour}")

    def gpu_accel_toggle(self):
        logging.info(f"Using GPU: {self.GPU.get()}")
        self.learn = load_learner(self.model_path, cpu=not self.GPU.get())

class ImageCaptureFrame(tk.Toplevel):
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
        
        # Custom method to execute when window is closed.
        # master.protocol('WM_DELETE_WINDOW', self.destructor)

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

        # self.protocol("WM_DELETE_WINDOW", self.destructor)

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
        self.output_path = filedialog.askdirectory() + '/'
        self.relpath.set('/' + os.path.relpath(self.output_path, os.getcwd()))
        count = len(os.listdir(self.output_path))
        logging.info("Change to {}".format(self.output_path))
        logging.info("current image count in output folder is {}".format(count))

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
        logging.info("closing GUI...")
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # close OpenCV windows


class Menu(tk.Tk):
    def __init__(self):
        """
        Main window, can select `Open Image Capture` to take more photos or `Open Production` to test the accuracy of the model
        TODO add trainer option
        """
        super().__init__()
        self.title("ENDG 411 Finger Digits")

        self.capture_btn = ttk.Button(self, text="Open Image Capture", command=self.open_capture)
        self.capture_btn.pack()

        self.production_btn = ttk.Button(self, text="Open Production", command=self.open_production)
        self.production_btn.pack()
        

    def open_capture(self):
        """Open capture window (a2)"""
        # Disable buttons so no other windows can open
        self.capture_btn.config(state="disabled")
        self.production_btn.config(state="disabled")

        self.capture_window = ImageCaptureFrame(self)
        self.capture_window.protocol("WM_DELETE_WINDOW", self.on_capture_close)

    def on_capture_close(self):
        """Method destroys capture child and reactivates menu buttons"""
        # Call window destructor
        self.capture_window.destructor()

        # Turn menu buttons back on
        self.capture_btn.config(state="normal")
        self.production_btn.config(state="normal")



    def open_production(self):
        """Open production window (a4)"""
        # Disable buttons so no other windows can open
        self.capture_btn.config(state="disabled")
        self.production_btn.config(state="disabled")
        # Turn menu buttons back on
        self.production_window = ProductionFrame(self)
        self.production_window.protocol("WM_DELETE_WINDOW", self.on_production_close)

    def on_production_close(self):
        """Method destroys production child and reactivates menu buttons"""
        # Call window destructor
        self.production_window.destructor()

        # Turn menu buttons back on
        self.capture_btn.config(state="normal")
        self.production_btn.config(state="normal")


        
def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    app = Menu()
    aw=400
    ah=300
    sw = app.winfo_screenwidth()
    sh = app.winfo_screenheight()
    app.geometry('%dx%d+%d+%d' % (aw, ah, (sw-aw)/2, (sh-ah)/2))
    app.mainloop()
        
if __name__ == '__main__':
    main()
