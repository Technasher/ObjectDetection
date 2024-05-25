from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilenames, askdirectory
import cv2
from PIL import Image, ImageTk
from cv2_enumerate_cameras import enumerate_cameras
from ObjectDetector import ObjectDetector

obj_detector = ObjectDetector('yolov8n.pt')
app = Tk()

img_filetypes = ['jpg', 'jpeg', 'png']
vid_filetypes = ['mp4', 'mov', 'mkv', 'm4v', 'avi']


def formating_extensions(extensions_list):
    return '*.' + ';*.'.join(extensions_list)


filetypes = [
    ('Картинки и видео', formating_extensions(img_filetypes + vid_filetypes)),
    ('Картинки', formating_extensions(img_filetypes)),
    ('Видео', formating_extensions(vid_filetypes))
]

# CAP_ANY     Any
# CAP_MSMF    Microsoft Media Foundation
# CAP_DSHOW   DirectShow
api_preference = cv2.CAP_DSHOW
camera_is_opened = False


def process_files():
    file_path_list = process_input_files_enter.get().split(',')
    progress_step = 100 / len(file_path_list)
    progress_bar['value'] = 0
    for file_path in file_path_list:
        name, extension = file_path.split('/')[-1].split('.')
        output_path = process_output_files_enter.get() + '/' + name
        if extension in img_filetypes:
            obj_detector.process_image(file_path.strip(), output_path + '.jpg')
        elif extension in vid_filetypes:
            obj_detector.process_video(file_path.strip(), output_path + '.mp4')
        progress_bar['value'] += progress_step


def get_input_files_from_filedialog():
    process_input_files_enter.delete(0, END)
    process_input_files_enter.insert(0, ', '.join(askopenfilenames(filetypes=filetypes)))


def get_output_dir():
    process_output_files_enter.delete(0, END)
    process_output_files_enter.insert(0, askdirectory())


def get_camera_list():
    return [f'{camera_info.index}: {camera_info.name}'
            for camera_info in enumerate_cameras(api_preference)]


def draw_frame(source):
    frame = next(source)
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    if camera_is_opened:
        label_widget.after(10, lambda: draw_frame(source))


def open_camera():
    global camera_is_opened
    if camera_list.get():
        if camera_is_opened:
            camera_is_opened = False
        else:
            camera_is_opened = True
            draw_frame(obj_detector.process_camera(int(camera_list.get().split(':')[0])))


label_widget = Label(app)
tool_menu = Frame(borderwidth=1, relief=SOLID)


label_widget.grid(row=0, column=0)
tool_menu.grid(row=0, column=1)

camera_menu = Frame(tool_menu, borderwidth=1, relief=SOLID)
camera_lable = Label(camera_menu, text='Камера')
camera_list = Combobox(camera_menu, values=get_camera_list(), width=50)
camera_list_lable = Label(camera_menu, text="Выберите камеру: ")
open_camera_btn = Button(camera_menu, text="Камера", command=open_camera)

process_menu = Frame(tool_menu, borderwidth=1, relief=SOLID)
process_lable = Label(process_menu, text='Обработка')
process_input_files_btn = Button(process_menu, text='Выбор',
                                 command=get_input_files_from_filedialog)
process_input_files_enter = Entry(process_menu, width=50)
process_input_files_lable = Label(process_menu, text='Выберите файлы: ')
process_output_files_btn = Button(process_menu, text='Выбор', command=get_output_dir)
process_output_files_enter = Entry(process_menu, width=50)
process_output_files_lable = Label(process_menu, text='Куда сохранить: ')
process_btn = Button(process_menu, text='Обработать', command=process_files)
progress_bar = Progressbar(process_menu, orient=HORIZONTAL, length=500, mode='determinate')

camera_menu.pack(fill='both')

camera_lable.grid(row=0, columnspan=2)
camera_list_lable.grid(row=1, column=0)
camera_list.grid(row=1, column=1)
open_camera_btn.grid(row=2, columnspan=2)

process_menu.pack(fill='both')

process_lable.grid(row=0, columnspan=3)
process_input_files_lable.grid(row=1, column=0)
process_input_files_enter.grid(row=1, column=1)
process_input_files_btn.grid(row=1, column=2)
process_output_files_lable.grid(row=2, column=0)
process_output_files_enter.grid(row=2, column=1)
process_output_files_btn.grid(row=2, column=2)
process_btn.grid(row=3, columnspan=3)
progress_bar.grid(row=4, columnspan=3)

app.mainloop()
