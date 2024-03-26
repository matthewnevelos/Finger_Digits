# Adapted from code found in this thread:
# https://stackoverflow.com/questions/32342935/using-opencv-with-tkinter

from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk
import cv2
import os
import logging
from tkinter import filedialog
import threading

class ProductionFrame(tk.Toplevel):
    pass

class ImageCaptureFrame(tk.Toplevel):
    def __init__(self, master):
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
                count (int): number used to create the next filename.
                current_image (PIL Image): current image displayed
                save_btn (ttk Button): press to save image
                burst_btn (ttk Button): press for burst mode
                browse_btn (ttk Button): press to change directory
                panel (ttk Label): to display image in frame
        """
        super().__init__(master)
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
        logging.info("closing GUI...")
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # close OpenCV windows
        self.destroy() # close the Tk window

class Menu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ENDG 411 Finger Digits")

        self.capture_btn = ttk.Button(self, text="Open Image Capture", command=self.open_capture)
        self.capture_btn.grid()

        self.production_btn = ttk.Button(self, text="Open Production", command=self.open_production)
        self.production_btn.grid()
        

    # Open capture window (a2)
    def open_capture(self):
        self.capture_btn.config(state="disabled")
        self.production_btn.config(state="disabled")

        self.capture_window = ImageCaptureFrame(self)
        self.capture_window.protocol("WM_DELETE_WINDOW", self.on_capture_close)

    # Call capture destructor
    def on_capture_close(self):
        # Call window destructor
        self.capture_window.destructor()

        # Turn menu buttons back on
        self.capture_btn.config(state="normal")
        self.production_btn.config(state="normal")


    # Open production window (a4)
    def open_production(self):
        self.capture_btn.config(state="disabled")
        self.production_btn.config(state="disabled")

        self.production_window = ProductionFrame(self)
        self.production_window.protocol("WM_DELETE_WINDOW", self.on_production_close)

    # Call production destructor
    def on_production_close(self):
        # Call window destructor
        self.production_window.destructor()

        # Turn menu buttons back on
        self.capture_btn.config(state="normal")
        self.production_btn.config(state="normal")

    
        
# def main():
#     logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    
#     # construct the argument parse and parse the arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-o", "--output", default="./",
#         help="path to output directory to store images (default: current folder")
#     args = parser.parse_args()
#     logging.info(f"saving images to {args.output}")

    
#     # start the app
#     logging.info("starting GUI...")
#     gui = tk.Tk() 
#     gui.title("Image Capture")  
#     ImageCaptureFrame(gui, args.output)
#     gui.mainloop()
        
def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    app = Menu()
    app.mainloop()
        
if __name__ == '__main__':
    main()
