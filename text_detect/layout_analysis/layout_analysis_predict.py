# import necessary packages
import os
import PIL.Image as PI
import numpy as np
import cv2
from wand.image import Image, Color

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('ocr')[0], 'ocr')


def resize_image(im, max_img_size=2048):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:     # 起到and的作用
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height

    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width

    # fixme 最多裁剪31个pixel 是否影响边缘效果
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


# processing letter by letter boxing
def process_letter(thresh, output):
    # assign the kernel size
    kernel = np.ones((2, 1), np.uint8)  # vertical
    # use closing morph operation then erode to narrow the image
    temp_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    # temp_img = cv2.erode(thresh,kernel,iterations=2)
    letter_img = cv2.erode(temp_img, kernel, iterations=1)

    # find contours
    (contours, _) = cv2.findContours(letter_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop in all the contour areas
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x-1, y-5), (x+w, y+h), (0, 255, 0), 1)
    return output


# processing letter by letter boxing
def process_word(thresh, output):
    # assign 2 rectangle kernel size 1 vertical and the other will be horizontal
    kernel = np.ones((2, 1), np.uint8)
    kernel2 = np.ones((1, 4), np.uint8)
    # use closing morph operation but fewer iterations than the letter then erode to narrow the image
    temp_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # temp_img = cv2.erode(thresh,kernel,iterations=2)
    word_img = cv2.dilate(temp_img, kernel2, iterations=1)

    (contours, _) = cv2.findContours(word_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x-1, y-5), (x+w, y+h), (0, 255, 0), 1)
    return output


# processing line by line boxing
def process_line(thresh, output):
    # assign a rectangle kernel size	1 vertical and the other will be horizontal
    kernel = np.ones((1, 10), np.uint8)
    kernel2 = np.ones((1, 8), np.uint8)
    # use closing morph operation but fewer iterations than the letter then erode to narrow the image
    temp_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2, iterations=2)
    # temp_img = cv2.erode(thresh, kernel, iterations=2)
    line_img = cv2.dilate(temp_img, kernel, iterations=5)

    (contours, _) = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    test_recs_all = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x-1, y-5), (x+w, y+h), (0, 255, 0), 1)
        recs = [x-1, y-5, x-1, y+h, x+w, y+h, x+w, y-5]
        test_recs_all.append(recs)
    text_recs_len = [len(contours)]
    test_recs_all = sorted(test_recs_all, key=lambda a: a[1])
    return test_recs_all, text_recs_len, output


# processing par by par boxing
def process_par(thresh, output):
    # assign a rectangle kernel size
    kernel = np.ones((5, 5), 'uint8')
    par_img = cv2.dilate(thresh, kernel, iterations=3)

    (contours, _) = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 1)
    return output


def main(img, im_name, max_img_size=2048):
    d_wight, d_height = resize_image(img, max_img_size=max_img_size)
    img = img.resize((d_wight, d_height), PI.BILINEAR).convert('RGB')
    image = np.array(img)
    img_all = np.expand_dims(image.copy(), axis=0)
    # output_letter = image.copy()
    # output_word = image.copy()
    output_line = image.copy()
    # output_par = image.copy()
    # hardcoded assigning of output images for the 3 input images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # clean the image using otsu method with the inversed binarized image
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # processing and writing the output
    # output_letter = process_letter(th, output_letter)
    # output_word = process_word(th, output_word)
    test_recs_all_line, text_recs_len_line, output_line = process_line(th, output_line)
    # output_par = process_par(th, output_par)
    # special case for the 5th output because margin with paragraph is just the 4th output with margin
    # cv2.imwrite(os.path.join(root_path, "LA_output/letter/%s_letter.jpg" % im_name), output_letter)
    # cv2.imwrite(os.path.join(root_path, "LA_output/word/%s_word.jpg" % im_name), output_word)
    cv2.imwrite(os.path.join(root_path, "LA_output/line/%s_line.jpg" % im_name), output_line)
    # cv2.imwrite(os.path.join(root_path, "LA_output/par/%s_par.jpg" % im_name), output_par)

    cv2.waitKey(0)
    return test_recs_all_line, text_recs_len_line, img_all


if __name__ == "__main__":
    sample_dir = os.path.join(root_path, 'sample/')
    sample_list = os.listdir(sample_dir)

    for i, sample in enumerate(sample_list):
        images = []
        img_path = os.path.join(sample_dir, sample)
        if sample[-4:] == '.pdf':
            with Image(filename=img_path, resolution=(200, 200)) as imgs:
                num_page = len(imgs.sequence)
                for i in range(num_page):
                    img = Image(image=imgs.sequence[i])
                    img.alpha_channel = False
                    img.background_color = Color('white')  # Set the background color
                    img = PI.fromarray(np.array(img), 'RGB')
                    im_name = sample[:-4] + '_p' + str(i)
                    images.append((img, im_name))
        else:
            im_name = sample[:-4]
            img = PI.open(img_path).convert('RGB')
            images = [(img, im_name)]
        for img, im_name in images:
            main(img, im_name, max_img_size=2048)
