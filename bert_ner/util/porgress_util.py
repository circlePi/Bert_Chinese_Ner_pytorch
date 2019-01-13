"""进度条"""

import sys


class ProgressBar(object):
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    # 初始化函数，需要知道总共的处理次数
    def __init__(self, epoch_size, batch_size, max_arrow=80):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.max_steps = round(epoch_size/batch_size)   # 总共处理次数 = round(epoch/batch_size)
        self.max_arrow = max_arrow   # 进度条的长度

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, train_acc, train_loss, f1, used_time, i):
        num_arrow = int(i * self.max_arrow / self.max_steps)  # 计算显示多少个'>'
        num_line = self.max_arrow - num_arrow  # 计算显示多少个'-'
        percent = i * 100.0 / self.max_steps  # 计算完成进度，格式为xx.xx%
        num_steps = self.batch_size * i    # 当前处理数据条数
        process_bar =  '%d'%num_steps + '/' + '%d'%self.epoch_size + '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + ' - train_acc ' + '%.4f'%train_acc + ' - train_loss '+ \
                       '%.4f' %train_loss + ' - f1 ' + '%.4f'% f1 + ' - time '+ '%.1fs'%used_time + '\r'
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()



