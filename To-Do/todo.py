import tkinter as tk
import platform

root = tk.Tk()
root.title("Sürüklenebilir Görev Listesi")


GRID_WIDTH = 150
GRID_HEIGHT = 80


canvas_width = 600
canvas_height = 400

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="#f0f5f9")
canvas.pack()

num_columns = canvas_width // GRID_WIDTH


if platform.system() == 'Darwin':
    RIGHT_CLICK = '<Button-2>'
else:
    RIGHT_CLICK = '<Button-3>'

class Task:
    def __init__(self, canvas, text, x=0, y=0):
        self.canvas = canvas
        self.text = text
        self.x = x
        self.y = y
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.rect = canvas.create_rectangle(
            self.x, self.y, self.x + self.width, self.y + self.height,
            fill="#a3d5d3", outline="#5eaaa8", width=2
        )
        self.label = canvas.create_text(
            self.x + 10, self.y + self.height / 2, anchor='w',
            text=self.text, font=("Helvetica", 14, "bold"), fill="#0f4c75"
        )
        
        
        self.canvas.tag_bind(self.rect, "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind(self.label, "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind(self.rect, "<B1-Motion>", self.on_move)
        self.canvas.tag_bind(self.label, "<B1-Motion>", self.on_move)
        self.canvas.tag_bind(self.rect, "<ButtonRelease-1>", self.on_release)
        self.canvas.tag_bind(self.label, "<ButtonRelease-1>", self.on_release)
        
        
        self.canvas.tag_bind(self.rect, RIGHT_CLICK, self.show_context_menu)
        self.canvas.tag_bind(self.label, RIGHT_CLICK, self.show_context_menu)
        
        
        self.menu = tk.Menu(root, tearoff=0)
        self.menu.add_command(label="Güncelle", command=self.update_task)
        self.menu.add_command(label="Sil", command=self.delete_task)
    
    def on_press(self, event):
        self.offset_x = self.canvas.canvasx(event.x) - self.x
        self.offset_y = self.canvas.canvasy(event.y) - self.y

    def on_move(self, event):
        self.x = self.canvas.canvasx(event.x) - self.offset_x
        self.y = self.canvas.canvasy(event.y) - self.offset_y
        self.canvas.coords(
            self.rect, self.x, self.y, self.x + self.width, self.y + self.height
        )
        self.canvas.coords(self.label, self.x + 10, self.y + self.height / 2)

    def on_release(self, event):
        
        grid_x = round(self.x / GRID_WIDTH) * GRID_WIDTH
        grid_y = round(self.y / GRID_HEIGHT) * GRID_HEIGHT
        self.x = grid_x
        self.y = grid_y
        self.canvas.coords(
            self.rect, self.x, self.y, self.x + self.width, self.y + self.height
        )
        self.canvas.coords(self.label, self.x + 10, self.y + self.height / 2)
    
    def show_context_menu(self, event):
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
    
    def delete_task(self):
        self.canvas.delete(self.rect)
        self.canvas.delete(self.label)
        tasks.remove(self)
    
    def update_task(self):
        
        update_window = tk.Toplevel(root)
        update_window.title("Görevi Güncelle")
        
        tk.Label(update_window, text="Yeni Görev:", font=("Helvetica", 12)).pack(pady=5)
        new_task_entry = tk.Entry(update_window, width=30, font=("Helvetica", 12))
        new_task_entry.pack(pady=5)
        new_task_entry.insert(0, self.text)
        
        def save_updated_task():
            new_text = new_task_entry.get()
            if new_text.strip():
                self.text = new_text
                self.canvas.itemconfigure(self.label, text=self.text)
                update_window.destroy()
            else:
                tk.messagebox.showwarning("Uyarı", "Lütfen bir görev girin.")
        
        tk.Button(update_window, text="Kaydet", command=save_updated_task, font=("Helvetica", 12)).pack(pady=10)
    
tasks = []

def add_task():
    task_text = entry.get()
    if task_text.strip():
        num_tasks = len(tasks)
        x = (num_tasks % num_columns) * GRID_WIDTH
        y = (num_tasks // num_columns) * GRID_HEIGHT
        new_task = Task(canvas, task_text, x, y)
        tasks.append(new_task)
        entry.delete(0, tk.END)
    else:
        tk.messagebox.showwarning("Uyarı", "Lütfen bir görev girin.")

entry = tk.Entry(root, width=50, font=("Helvetica", 12))
entry.pack(pady=10)

add_button = tk.Button(root, text="Görev Ekle", width=48, command=add_task, font=("Helvetica", 12))
add_button.pack()

root.mainloop()
