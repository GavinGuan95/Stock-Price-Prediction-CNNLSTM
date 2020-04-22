import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from trader import Trader
import os

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
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
        self.trader = Trader(self.config, self.valid_data_loader)
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
            # print("training batch_idx: {}, target: {}, data: {}".format(batch_idx, target, data))
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
            val_log, conf_matrix = self._valid_epoch(epoch)
            val_log["confusion matrix"] = conf_matrix
            self.trader.plot_ret()
            buy_and_hold_sharpe, regression_sharpe = self.trader.get_sharpe()
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            if os.path.exists("results.npz"):
                with np.load("results.npz") as result:
                    mse, sharpe, reg_binary_pred, F_1_score, MAPE = [result[i] for i in ('mse_loss', 'regression_sharpe', 'regression_binary_pred', 'F_1_score','MAPE')]

                np.savez("results.npz", mse_loss=min(mse, val_log["loss"]), regression_sharpe=max(sharpe, regression_sharpe),
                         regression_binary_pred=max(reg_binary_pred, val_log["regression_binary_pred"]),MAPE = min(MAPE,val_log["MAPE"]),
                         F_1_score=max(F_1_score, val_log["f1_score"]),conf_mtx = conf_matrix)
            else:
                np.savez("results.npz", mse_loss=val_log["loss"], regression_sharpe=regression_sharpe, regression_binary_pred=val_log["regression_binary_pred"], F_1_score=val_log["f1_score"],MAPE = val_log["MAPE"],conf_mtx = conf_matrix)
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
        print("starting _valid_epoch")
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                # print("validation batch_idx: {} ,target: {}, data: {}".format(batch_idx, target, data))
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
        return self.valid_metrics.result(),self.valid_metrics.get_conf_mtx()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
