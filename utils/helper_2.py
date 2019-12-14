import datetime
import os


def LOG(message, log_file):
    ts = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    msg = "[%s] %s" % (ts, message)

    with open(log_file, "a") as fp:
        fp.write(msg + "\n")
    print(msg)


def log_variables_to_board(epoch_losses, losses, losses_name, phrase, folder, epoch, log_file, writer):
    # global writer
    # LOG(phrase + " epoch    " + str(epoch+1) + ":", log_file)
    for e_loss, l, l_n in zip(epoch_losses, losses, losses_name):
        e_loss.append(l)
        writer.add_scalar(folder + os.sep + "data" + os.sep + l_n, l, epoch)
        # LOG("          " + l_n + ": "+ str(l), log_file)

    # LOG("\n", log_file)


def save_csv_logging(csv_checkpoint, epoch, lr, losses, val_losses, total_loss, total_val_loss, csv_path, log_file):
    try:
        csv_checkpoint.loc[len(csv_checkpoint)] = [str(datetime.datetime.now()), epoch, lr,
                                                   losses[0], losses[1], losses[2],
                                                   val_losses[0], val_losses[1], val_losses[2],
                                                   val_losses[0], val_losses[1], val_losses[2],
                                                   total_loss, total_val_loss]
        csv_checkpoint.to_csv(csv_path, index=False)

    except:
        LOG("Error when saving csv files! [tip]: Please check csv column names.", log_file)

        # print(csv_checkpoint.columns)
    return csv_checkpoint
