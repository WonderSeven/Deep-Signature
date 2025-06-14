import os
import json
import numpy as np


def process_each_file(file_path):
    with open(file_path, 'r') as textFile:
        context = textFile.readlines()
        val_line, test_line = context[-2:]
        val_line, test_line = val_line.strip(), test_line.strip()

        val_line_start = val_line.index('Acc:') + len('Acc:')
        cur_context = val_line[val_line_start+1: val_line.index(',', val_line_start)].strip()
        val_acc = float(cur_context)
        val_line_start = val_line.index('Rec:') + len('Rec:')
        cur_context = val_line[val_line_start+1: val_line.index(',', val_line_start)].strip()
        val_rec = float(cur_context) #* 100.
        val_line_start = val_line.index('F1:') + len('F1:')
        cur_context = val_line[val_line_start+1: -1].strip()
        val_f1 = float(cur_context) #* 100.

        test_line_start = test_line.index('Acc:') + len('Acc:')
        cur_context = test_line[test_line_start+1: test_line.index(',', test_line_start)].strip()
        test_acc = float(cur_context)
        test_line_start = test_line.index('Rec:') + len('Rec:')
        cur_context = test_line[test_line_start+1: test_line.index(',', test_line_start)].strip()
        test_rec = float(cur_context) # * 100.
        test_line_start = test_line.index('F1:') + len('F1:')
        cur_context = test_line[test_line_start+1: -1].strip()
        test_f1 = float(cur_context) #* 100.

    return val_acc, val_rec, val_f1, test_acc, test_rec, test_f1


def process_each_test_file(file_path):
    with open(file_path, 'r') as textFile:
        context = textFile.readlines()
        test_line = context[-1:][0]

        test_line_start = test_line.index('Acc:') + len('Acc:')
        cur_context = test_line[test_line_start+1: test_line.index(',', test_line_start)].strip()
        test_acc = float(cur_context)

        test_line_start = test_line.index('Rec:') + len('Rec:')
        cur_context = test_line[test_line_start+1: test_line.index(',', test_line_start)].strip()
        test_rec = float(cur_context)

        test_line_start = test_line.index('F1:') + len('F1:')
        cur_context = test_line[test_line_start+1: -1].strip()
        test_f1 = float(cur_context)

    return test_acc, test_rec, test_f1


if __name__ == '__main__':
    file_root = './logs/EGFR_all_5G_5F/DeepSigL3_Local_C50_LN_LR1e-6_PW2.5'

    files = os.listdir(file_root)
    files.sort()
    precision = 3

    val_acc_list, val_rec_list, val_f1_list = [], [], []
    test_acc_list, test_rec_list, test_f1_list = [], [], []
    file_nums = 0
    for file_name in files:
        # filter out non-relative files
        if file_name.startswith('train') and file_name.endswith('.txt'):
            file_path = os.path.join(file_root, file_name)
            print('Reading file:{}'.format(file_path))
            file_nums += 1

            val_acc, val_rec, val_f1, test_acc, test_rec, test_f1 = process_each_file(file_path)
            # print('Val acc:{}, rec:{:.3f}, f1:{:.3f} | Test acc:{}, rec:{:.3f}, f1:{:.3f}'.format(val_acc, val_rec, val_f1, test_acc, test_rec, test_f1))
            val_acc_list.append(val_acc)
            val_rec_list.append(val_rec)
            val_f1_list.append(val_f1)
            test_acc_list.append(test_acc)
            test_rec_list.append(test_rec)
            test_f1_list.append(test_f1)

            # if file_nums == 4:
            #      break

    val_acc_mean = np.around(np.mean(val_acc_list), decimals=precision)
    val_rec_mean = np.around(np.mean(val_rec_list), decimals=precision)
    val_f1_mean = np.around(np.mean(val_f1_list), decimals=precision)
    test_acc_mean = np.around(np.mean(test_acc_list), decimals=precision)
    test_rec_mean = np.around(np.mean(test_rec_list), decimals=precision)
    test_f1_mean = np.around(np.mean(test_f1_list), decimals=precision)

    print('=========================================')
    print('Total Val Acc:{:.3f}±{:.3f}, Rec:{:.3f}±{:.3f}, F1:{:.3f}±{:.3f} %'.format(val_acc_mean, np.std(val_acc_list),
                                                                                      val_rec_mean, np.std(val_rec_list),
                                                                                      val_f1_mean,np.std(val_f1_list)))
    print('Total Test Acc:{:.3f}±{:.3f}, Rec:{:.3f}±{:.3f}, F1:{:.3f}±{:.3f} %'.format(test_acc_mean, np.std(test_acc_list),
                                                                                      test_rec_mean, np.std(test_rec_list),
                                                                                      test_f1_mean,np.std(test_f1_list)))

    """"
    for file_name in files:
        # filter out non-relative files
        if file_name.startswith('test') and file_name.endswith('.txt'):
            file_path = os.path.join(file_root, file_name)
            print('Reading file:{}'.format(file_path))
            file_nums += 1

            test_acc, test_rec, test_f1 = process_each_test_file(file_path)
            print('Test:{}'.format(test_acc))
            test_acc_list.append(test_acc)
            test_rec_list.append(test_rec)
            test_f1_list.append(test_f1)

            # if file_nums == 4:
            #      break

    test_acc_mean = np.around(np.mean(test_acc_list), decimals=precision)
    test_rec_mean = np.around(np.mean(test_rec_list), decimals=precision)
    test_f1_mean = np.around(np.mean(test_f1_list), decimals=precision)

    print('=========================================')
    print('Total Test Acc:{:.3f}±{:.3f}, Rec:{:.3f}±{:.3f}, F1:{:.3f}±{:.3f} %'.format(test_acc_mean, np.var(test_acc_list),
                                                                                      test_rec_mean, np.var(test_rec_list),
                                                                                      test_f1_mean,np.var(test_f1_list)))
    """