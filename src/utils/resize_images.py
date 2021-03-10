import glob
from tqdm import tqdm
import os
from PIL import Image, ImageFile
from joblib import delayed, Parallel

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_image(filepath, output_folder, resize):
    base_name = os.path.basename(filepath)
    img = Image.open(filepath)
    outpath = os.path.join(output_folder, base_name)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )
    img.save(outpath)

if __name__ == "__main__":
    input_train_folder = '../DATA/train/'
    input_test_folder = '../DATA/test/'

    os.mkdir('../DATA/train224')
    os.mkdir('../DATA/test224')

    output_train_folder = '../DATA/train224'
    output_test_folder = '../DATA/test224'
    train_images = glob.glob(os.path.join(input_train_folder, "*.jpg"))
    test_images = glob.glob(os.path.join(input_test_folder, "*.jpg"))

    Parallel(n_jobs=20)(
            delayed(resize_image)(
                i,
                output_train_folder,
                (224, 224)
            ) for i in tqdm(train_images)
        )

    Parallel(n_jobs=20)(
            delayed(resize_image)(
                i,
                output_test_folder,
                (224, 224)
            ) for i in tqdm(test_images)
        )
    
    """
    image_list = [train_images, test_images]
    output_list = [output_train_folder, output_test_folder]
    for images, output_folder in zip(image_list, output_list):
        Parallel(n_jobs=20)(
            delayed(resize_image)(
                i,
                output_folder,
                (224, 224)
            ) for i in tqdm(images)
        )"""