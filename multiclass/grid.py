import numpy as np
import os
import cv2

def create_grid(images_path,label): 
    images_in_row = []
    images_all = [] 
    for image_path in images_path:
        image = cv2.imread(f'viz/{image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = cv2.resize(image,(360,200))
        images_in_row.append(image)
        if len(images_in_row) == 6:
            images_in_row = np.concatenate(images_in_row, axis=1)
            images_in_row_shape = images_in_row.shape
            images_all.append(images_in_row)
            images_in_row = []
            
        if len(images_all)==8:
            break

    if len(images_all)>0:
        images_all = np.concatenate(images_all, axis=0)
        images_all = cv2.cvtColor(images_all, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{label}_images.jpg',images_all)
        
        
if __name__ == '__main__':
    images = os.listdir('viz')
    bad_crops = [image  for image in images if '__cobbling.jpg' in image]
    #good_crops = [image for image in images if '_no_cobbling.jpg' in image]
    create_grid(bad_crops,'cobbling')
    #create_grid(good_crops,'no_cobbling')