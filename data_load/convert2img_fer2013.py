import csv
import os
from PIL import Image
import numpy as np

database_path = r'/media/yi/harddrive/data/FER2013/all/fer2013'
datasets_path = r'/media/yi/harddrive/data/FER2013/all/fer2013_img'
csv_file = os.path.join(database_path, 'fer2013.csv')
train_csv = os.path.join(datasets_path, 'train.csv')
val_csv = os.path.join(datasets_path, 'val.csv')
test_csv = os.path.join(datasets_path, 'test.csv')


if __name__ == "__main__":

    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        rows = [row for row in csvr]

        trn = [row[:-1] for row in rows if row[-1] == 'Training']
        csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
        print(len(trn))

        val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
        csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
        print(len(val))

        tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + tst)
        print(len(tst))

    print("split train, valid, test dataset done")


    datasets_path = r'/media/yi/harddrive/data/FER2013/all/fer2013_img'
    train_csv = os.path.join(datasets_path, 'train.csv')
    val_csv = os.path.join(datasets_path, 'val.csv')
    test_csv = os.path.join(datasets_path, 'test.csv')

    train_set = os.path.join(datasets_path, 'train')
    val_set = os.path.join(datasets_path, 'val')
    test_set = os.path.join(datasets_path, 'test')

    for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv), (test_set, test_csv)]:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        num = 1
        with open(csv_file) as f:
            csvr = csv.reader(f)
            header = next(csvr)
            for i, (label, pixel) in enumerate(csvr):
                pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
                subfolder = os.path.join(save_path, label)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                im = Image.fromarray(pixel).convert('L')
                image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
                print(image_name)
                im.save(image_name)

    print("done")