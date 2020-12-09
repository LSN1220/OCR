import os

job_name = 'job_crnn_1208'

# data dir
fonts_dir = 'fonts'
bg_dir = 'bg'
char_dir = 'chars'
uni_chars_file = 'jpn.txt'
evl_dir = 'evl_data'
models_dir = 'chekcpoints'

job_dir = os.path.join(
    os.path.abspath(os.path.dirname(__file__)).split('text_recognize')[0], 'text_recognize')
tran_datas_dir = os.path.join(job_dir, 'data')
# founts
fonts_dir_abs = os.path.join(tran_datas_dir, fonts_dir)
fonts_files = [os.path.join(fonts_dir_abs, fonts_item) for fonts_item in os.listdir(fonts_dir_abs)]
# chars
char_dir_abs = os.path.join(tran_datas_dir, char_dir)
uni_chars_file = os.path.join(char_dir_abs, uni_chars_file)
# bg
bg_dir_abs = os.path.join(tran_datas_dir, bg_dir)
bg_files = [os.path.join(bg_dir_abs, bg_item) for bg_item in os.listdir(bg_dir_abs)]
# data generator output
job_output_path = os.path.join(job_dir, job_name)

