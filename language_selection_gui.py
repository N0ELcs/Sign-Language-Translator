# language_selection_gui.py
import tkinter as tk
from tkinter import ttk

def select_language(languages):
    def on_language_selected(event):
        global selected_language
        selected_language = language_combobox.get()
        root.destroy()

    root = tk.Tk()
    root.title("Select Sign Language")
    selected_language = tk.StringVar()
    language_combobox = ttk.Combobox(root, textvariable=selected_language, values=languages)
    language_combobox.pack()
    language_combobox.bind('<<ComboboxSelected>>', on_language_selected)
    root.mainloop()
    return selected_language.get()
