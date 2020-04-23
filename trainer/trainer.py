import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from trader import Trader
from sklearn.metrics import roc_auc_score
import os

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, df,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # TODO [Gavin]: tensorboard visualization commented out
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.trader = Trader(self.config, self.valid_data_loader, df)
        # self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # lowest_mse_loss = float("inf")
        # high_reg_sharpe = -float("inf")
        # high_bi_pred = -float("inf")
        # high_f1 = -float("inf")
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            # TODO [Gavin]: tensorboard visualization commented out
            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # training loss is mse loss
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                # regression binary pred was performed in here
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # TODO [Gavin]: tensorboard visualization commented out
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, conf_matrix, comp_mtx, sign_mtx = self._valid_epoch(epoch)
            # Have to calcluate F-1 score in here
            TP = conf_matrix[0][1]
            TN = conf_matrix[0][0]
            FP = conf_matrix[1][1]
            FN = conf_matrix[1][0]

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f_1_score = 2 * precision * recall / (precision + recall)


            # calcualte MAPE in here
            output_accu = comp_mtx[:,0]
            target_accu = comp_mtx[:,1]
            N = output_accu.shape[0]
            mape = np.sum(np.absolute((output_accu - target_accu) / target_accu)) / N


            # calculate roc_auc in here
            output_sign = sign_mtx[:, 0]
            target_sign = sign_mtx[:, 1]
            r_c_score = roc_auc_score(target_sign, output_sign)

            # log the calcuated values
            val_log['f1_score'] = f_1_score
            val_log['precision'] = precision
            val_log['recall'] = recall
            val_log["confusion_matrix"] = conf_matrix
            val_log["MAPE"] = mape
            val_log["roc_auc"] = r_c_score



            buy_and_hold_sharpe, regression_sharpe = self.trader.get_sharpe()
            log.update(**{'val_'+k : v for k, v in val_log.items()})

            if os.path.exists("results.npz"):
                with np.load("results.npz") as result:
                    mse, reg_binary_pred, F_1_score, precision, recall, MAPE,r_c_score = [result[i] for i in ('mse_loss', 'regression_binary_pred', 'F_1_score','precision','recall','MAPE','roc_auc')]
                # F_1_score has only 1 element, but somehow the numpy array size is zero, retrieve that value
                self.trader.plot_ret(F_1_score.max()>f_1_score)
                np.savez("results.npz",
                         mse_loss=min(mse, val_log["loss"]),
                         regression_binary_pred=max(reg_binary_pred, val_log["regression_binary_pred"]),
                         MAPE = min(MAPE,val_log["MAPE"]),
                         F_1_score=max(F_1_score, f_1_score),
                         precision= precision if F_1_score>f_1_score else val_log['precision'],
                         recall=recall if F_1_score>f_1_score else val_log['recall'],
                         conf_mtx = conf_matrix if F_1_score>f_1_score else val_log["confusion_matrix"],
                         roc_auc=max(r_c_score,val_log["roc_auc"]))
            else:
                np.savez("results.npz",
                         mse_loss=val_log["loss"],
                         regression_binary_pred=val_log["regression_binary_pred"],
                         F_1_score=f_1_score,
                         precision = precision,
                         recall = recall,
                         MAPE = mape,
                         conf_mtx = conf_matrix,
                         roc_auc = r_c_score)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        all_validation_output = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                # TODO [Gavin]: tensorboard visualization commented out
                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # TODO [Gavin]: tensorboard visualization commented out
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                all_validation_output.append(output)
        self.trader.calc_return(all_validation_output)
        # TODO [Gavin]: tensorboard visualization commented out
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(),self.valid_metrics.get_conf_mtx(),self.valid_metrics.get_comp_mtx(),self.valid_metrics.get_sign_mtx()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
