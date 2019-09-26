import torch
import matplotlib.pyplot as plt

Cha_Learn_2016_label_names = [
    'age_1', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7', 'age_8', 'age_9', 'age_10', 
    'age_11', 'age_12', 'age_13', 'age_14', 'age_15', 'age_16', 'age_17', 'age_18', 'age_19', 'age_20', 
    'age_21', 'age_22', 'age_23', 'age_24', 'age_25', 'age_26', 'age_27', 'age_28', 'age_29', 'age_30', 
    'age_31', 'age_32', 'age_33', 'age_34', 'age_35', 'age_36', 'age_37', 'age_38', 'age_39', 'age_40', 
    'age_41', 'age_42', 'age_43', 'age_44', 'age_45', 'age_46', 'age_47', 'age_48', 'age_49', 'age_50', 
    'age_51', 'age_52', 'age_53', 'age_54', 'age_55', 'age_56', 'age_57', 'age_58', 'age_59', 'age_60', 
    'age_61', 'age_62', 'age_63', 'age_64', 'age_65', 'age_66', 'age_67', 'age_68', 'age_69', 'age_70', 
    'age_71', 'age_72', 'age_73', 'age_74', 'age_75', 'age_76', 'age_77', 'age_78', 'age_79', 'age_80', 
    'age_81', 'age_82', 'age_83', 'age_84', 'age_85', 'age_86', 'age_87', 'age_88', 'age_89', 'age_90', 
    'age_91', 'age_92', 'age_93', 'age_94', 'age_95', 'age_96', 'age_97', 'age_98', 'age_99', 'age_100'
]

def plot_images(images, cls_true, name):
    assert len(images) == len(cls_true) == 36
    # Create figure with sub-plots.
    fig, axes = plt.subplots(6, 6)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i, :, :, :], interpolation='spline16')

        # get its equivalent class name
        if name == 'cifar10':
            cls_true_name = Cha_Learn_2016_label_names[cls_true[i]]
        else:
            cls_true_name = Cha_Learn_2016_label_names[cls_true[i]]

        xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])  
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def convert_tensor_to_image(img_tensor, labels):
    '''
    convert tensor to image and show it
    '''
    X = img_tensor.cpu().numpy()
    X = np.transpose(X, [0, 2, 3, 1])
    plot_images(X, labels, "convert_tensor_to_image")
    return 0

def convert_to_onehot_tensor(y, nb_digits):
    # y = [1,2,3,4]
    y = y.reshape([len(y), 1])
    # y = [[1],[2],[3],[4]]
    batch_size = y.size()[0]
    y_onehot = torch.cuda.LongTensor(batch_size, nb_digits)
    # y_onehot = torch.cuda.FloatTensor(batch_size, nb_digits)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot
