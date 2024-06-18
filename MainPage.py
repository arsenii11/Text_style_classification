import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import chardet  # You might need to install this package via pip
from PyPDF2 import PdfReader  # You might need to install this package via pip


def submit():
    # Retrieve the selected style and text from the text input
    selected_style = style_var.get()
    input_text = text_input.get("1.0", tk.END)
    print(f"Selected Narrative Style: {selected_style}")
    print(f"Input Text: {input_text.strip()}")  # Use strip to remove extra newline


def load_file():
    # Open file dialog to select a file
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf")])
    if file_path:
        try:
            if file_path.endswith('.pdf'):
                # Read PDF file content
                reader = PdfReader(file_path)
                file_content = ""
                for page in reader.pages:
                    file_content += page.extract_text()
            else:
                # Detect file encoding
                with open(file_path, 'rb') as file:
                    raw_data = file.read()
                    result = chardet.detect(raw_data)
                    file_encoding = result['encoding']

                # Read file content with detected encoding
                with open(file_path, 'r', encoding=file_encoding) as file:
                    file_content = file.read()

            text_input.delete("1.0", tk.END)  # Clear the text input field
            text_input.insert(tk.END, file_content)  # Insert the file content into the text input field
        except Exception as e:
            print(f"Error reading file: {e}")


# Create the main window
root = tk.Tk()
root.title("Narrative Style Interface")

# Ensure the window has a larger size
root.geometry('1200x800')  # Adjust the size to be larger

# Label for selecting the narrative style
label = tk.Label(root, text="Select a narrative style:")
label.pack(pady=10)

# List of narrative styles
styles = ["Official-Business", "Journalistic", "Scientific", "Literary"]

# Variable to store the selected style
style_var = tk.StringVar(value=styles[0])

# Create the dropdown menu
style_dropdown = ttk.Combobox(root, textvariable=style_var, values=styles, state="readonly")
style_dropdown.pack(pady=5)

# Button to load a file
load_file_btn = tk.Button(root, text="Load File", command=load_file)
load_file_btn.pack(pady=10)

# Text input field for narrative text
text_input = tk.Text(root, height=30, width=100)  # Adjusted size to ensure it's visible
text_input.pack(pady=10)

# Button to submit the inputs
submit_btn = tk.Button(root, text="Enter", command=submit)
submit_btn.pack(pady=20)

# Start the Tkinter main loop
root.mainloop()
