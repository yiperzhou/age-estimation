import os
import datetime

def LOG(message, logFile):
    ts = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    msg = "[%s] %s" % (ts, message)

    with open(logFile, "a") as fp:
        fp.write(msg + "\n")

    print(msg)


def log_variables_to_board(epoch_losses, losses, losses_name, epoch_accs, accs, accs_name, phrase, folder, epoch, logFile, writer):
    '''
    
    '''
    # global writer
    # LOG(phrase + " epoch    " + str(epoch+1) + ":", logFile)
    for e_loss, l, l_n in zip(epoch_losses, losses, losses_name):
        e_loss.append(l)
        writer.add_scalar(folder + os.sep + "data" + os.sep + l_n, l, epoch)
        # LOG("          " + l_n + ": "+ str(l), logFile)

    # LOG("\n", logFile)

    for e_acc, ac, ac_n in zip(epoch_accs, accs, accs_name):
        e_acc.append(ac)
        writer.add_scalar(folder + os.sep + "data" + os.sep + ac_n, ac, epoch)
        # LOG("          " + ac_n + ": "+ str(ac), logFile)
    # LOG("---------------", logFile)
    


def save_csv_logging(csv_checkpoint, epoch, lr, losses, val_losses, accs, val_accs, total_loss, total_val_loss, csv_path, logFile):
    '''

    '''
    try:
        csv_checkpoint.loc[len(csv_checkpoint)] = [str(datetime.datetime.now()), epoch, lr, 
                                                    accs[0], accs[1], losses[0], losses[1], losses[2],
                                                    val_accs[0], val_accs[1], val_losses[0], val_losses[1], val_losses[2], 
                                                    total_loss, total_val_loss]
        csv_checkpoint.to_csv(csv_path, index=False)

    except:
        LOG("Error when saving csv files! [tip]: Please check csv column names.", logFile)

        # print(csv_checkpoint.columns)
    return csv_checkpoint
