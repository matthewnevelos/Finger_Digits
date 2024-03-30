# Adapted from code found in this thread:
# https://stackoverflow.com/questions/32342935/using-opencv-with-tkinter
import tkinter as tk
from tkinter import ttk
import logging
from finger_digit.windows import ImageCaptureWindow, ProductionWindow, TrainingWindow

# TODO add model training

class Menu(tk.Tk):
    def __init__(self):
        """
        Main window, can select: `Open Image Capture` to take more photos
                                 `Open Model Training` to train the model
                                 `Open Production` to test the accuracy of the model
        """
        super().__init__()
        self.title("ENDG 411 Finger Digit Menu")
        logging.info("Starting Digit Wizard")

        self.capture_btn = ttk.Button(self, text="Open Capture", command=lambda: self.open_window(ImageCaptureWindow, "Opening image capture window"))
        self.capture_btn.pack()

        self.train_btn = ttk.Button(self, text="Open Training", command=lambda: self.open_window(TrainingWindow, "Opening model training window"))
        self.train_btn.pack()

        self.production_btn = ttk.Button(self, text="Open Production", command=lambda: self.open_window(ProductionWindow, "Opening production digit reading"))
        self.production_btn.pack()

        self.buttons = [self.capture_btn, self.train_btn, self.production_btn]
        

    def open_window(self, window, opening_text):
        """
        Create child window from a finger_digits.apps Toplevel class
        Parameters:
            window (tk.Toplevel): The frame which will be made
            opening_text (str): Text logged to console when window is opened
        """
        logging.info(opening_text)

        self.disable_btns()

        # Create new window
        window = window(self)
        # Call the windows destructor and enable buttons in the Menu window when the child window is closed 
        window.protocol("WM_DELETE_WINDOW", lambda: (window.destructor(), self.enable_btns())) #must be a better way to do these 2 commands at once?


    def disable_btns(self):        
        # Disable buttons so no other windows can open
        for x in self.buttons:
            x.config(state="disabled")

    def enable_btns(self):
        # Enable buttons so the window can be opened again
        for x in self.buttons:
            x.config(state="normal")


    def destroy(self):
        """Kill parent window"""
        logging.info("Closing Finger Digit Wizard")
        super().destroy()
        
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
