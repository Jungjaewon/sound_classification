import os
import time
import datetime
import torch
import torch.nn as nn
import glob
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from sklearn.metrics import confusion_matrix
from model import ResAttentionModel
from loss import FocalLoss

matplotlib.use('Agg')


class Solver(object):

    def __init__(self, config, train_data_loader, test_data_loader):
        """Initialize configurations."""
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        self.sample_dim   = config['MODEL_CONFIG']['SAMPLE_DIM']
        assert self.sample_dim in [128]

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.lr            = float(config['TRAINING_CONFIG']['LR'])
        self.lambda_cls = config['TRAINING_CONFIG']['LAMBDA_CLS']
        self.lambda_focal = config['TRAINING_CONFIG']['LAMBDA_FOCAL']

        self.train_loss_tracker = list()
        self.train_acc_tracker = list()

        self.test_loss_tracker = list()
        self.test_acc_tracker = list()

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.label_list = ["air_conditioner", "car_horn", "children_playing",
                           "dog_bark", "drilling", "engine_idling", "gun_shot",
                           "jackhammer", "siren", "street_music"]
        self.data_dir = config['TRAINING_CONFIG']['DATA_DIR']

        #self.criterion = nn.CrossEntropyLoss()
        print('alpha : ', self.get_alpha())
        self.criterion = FocalLoss(gamma=2, alpha=self.get_alpha())
        self.max_acc = 0.0
        self.save_flag = False

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD'] == 'True'

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = os.path.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = os.path.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step  = config['TRAINING_CONFIG']['LR_DECAY_STEP']
        self.lr_update_step  = config['TRAINING_CONFIG']['LR_UPDATE_STEP']

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        self.model = ResAttentionModel(num_label=10).to(self.gpu)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, (self.beta1, self.beta2))

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def restore_model(self):

        ckpt_list = glob.glob(osp.join(self.model_dir, '*.ckpt'))

        if len(ckpt_list) == 0:
            return 0

        ckpt_list = [int(x.split(os.sep)[-1].split('.')[0]) for x in ckpt_list]
        ckpt_list.sort()
        epoch = ckpt_list[-1]
        model_path = os.path.join(self.model_dir, '{}.ckpt'.format(epoch))
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model.to(self.gpu)
        return epoch

    def get_alpha(self):
        total_data = 0
        alpha_list = list()
        for label in self.label_list:
            label_data_cnt = len(glob.glob(osp.join(self.data_dir, f'{label}_train', '*.npy')))
            total_data += label_data_cnt
            alpha_list.append(label_data_cnt)
        return list(map(lambda x: x / total_data, alpha_list))

    def save_test_figures(self, x_axis, loss_source, acc_source, label="test"):

        plt.figure()
        plt.title("test_loss_acc")
        plt.xlabel('Epoch')
        plt.ylabel('')
        plt.plot(x_axis, loss_source, color='r', label=f"{label} loss")
        plt.plot(x_axis, acc_source, color='g', label=f"{label} acc")
        plt.legend()
        plt.legend(loc="lower right")
        plt.savefig(osp.join(self.sample_dir, f'{label}_loss_acc_graph.png'), dpi=150)
        plt.close()

    def plotingConfusionMatrix(self, target, pred, class_list, epoch, save_dir, normalize=True, title=None, cmap=plt.cm.Blues, prefix=None):

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        img_name = 'epoch_{}_confusionMatrix.png'.format(str(epoch).zfill(3))
        if isinstance(prefix, str):
            img_name = prefix + img_name

        # Compute confusion matrix
        cm = confusion_matrix(target, pred)
        cm1 = cm
        class_list = np.array(class_list)

        if normalize:
            cm = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100, 0)
        else:
            print('Confusion matrix, without normalization')

        fig, ax = plt.subplots(figsize=(15, 15))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=class_list, yticklabels=class_list,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                num = format(cm[i, j], fmt)
                num = num.split(".")[0]
                ax.text(j, i, str(cm1[i, j]) + " (" + num + "%)",
                        ha="center", va="center",
                        #color="black", fontsize=8)
                        color="white" if cm[i, j] > thresh else "black", fontsize=12)
        #fig.tight_layout()
        fig.savefig(osp.join(save_dir, img_name), dpi=300)
        plt.close()
        return ax

    def train(self):

        # Set data loader.
        train_data_loader = self.train_data_loader
        iterations = len(self.train_data_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        train_data_iter = iter(train_data_loader)
        fixed_batch, fixed_label = next(train_data_iter)
        fixed_batch, fixed_label = fixed_batch.to(self.gpu), fixed_label.to(self.gpu)
        fixed_label = fixed_label.squeeze()

        start_epoch = self.restore_model()
        start_time = time.time()

        # =================================================================================== #
        #                             2. Train the model                                      #
        # =================================================================================== #

        print('Start training...')
        for e in range(start_epoch, self.epoch):

            training_loss_epoch = list()
            training_acc_epoch = list()
            for i in range(iterations):
                try:
                    batch, label = next(train_data_iter)
                except:
                    train_data_iter = iter(train_data_loader)
                    batch, label = next(train_data_iter)

                batch, label = batch.to(self.gpu), label.to(self.gpu)
                label = label.squeeze()


                prediction = self.model(batch)
                training_loss = self.criterion(prediction, label) * self.lambda_cls

                _, prd_idx = torch.max(prediction, 1)
                correct = (prd_idx == label).sum().cpu().item()

                if torch.isnan(training_loss):
                    raise Exception('loss_fake is nan at {}'.format(e * iterations + (i + 1)))

                self.reset_grad()
                training_loss.backward()
                self.optimizer.step()

                loss_dict = dict()
                loss_dict['training_loss'] = training_loss.item()
                training_loss_epoch.append(training_loss.item())
                training_acc_epoch.append(correct / float(self.batch_size))

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            self.train_loss_tracker.append(sum(training_loss_epoch) / float(len(training_loss_epoch)))
            self.train_acc_tracker.append(sum(training_acc_epoch) / float(len(training_acc_epoch)))

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():
                    prediction = self.model(fixed_batch)
                    _, prediction = torch.max(prediction, 1)
                    total = fixed_label.size(0)
                    correct = (prediction == fixed_label).sum().item()
                    print('Accuracy of the net on the fixedBatch : {:.4f}'.format((100 * correct / total)))

                if (e + 1) % self.save_step == 0:
                    self.test(e + 1, self.test_data_loader, 'test')

            """
            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                model_path = os.path.join(self.model_dir, '{}.ckpt'.format(e + 1))
                torch.save(self.model.state_dict(), model_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))
            """

            if (e + 1) % self.lr_update_step == 0 and self.lr_decay_step > 0:
                pass

        self.save_test_figures(list(range(1, len(self.train_loss_tracker) + 1)), self.train_loss_tracker, self.train_acc_tracker, label="train")
        self.save_test_figures(list(range(1, len(self.test_loss_tracker) + 1)), self.test_loss_tracker, self.test_acc_tracker, label="test")
        print('Training is finished')

    def test(self, epoch, data_loader, mode):

        self.model = self.model.eval()

        correct = 0
        total = 0
        pred_idx_list = list()
        target_list = list()
        testing_loss_list = list()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                test_data, label = data
                test_data = test_data.to(self.gpu)
                label = label.to(self.gpu)
                #label = label.squeeze()

                prediction = self.model(test_data)
                testing_loss = self.criterion(prediction, label) * self.lambda_cls
                testing_loss_list.append(testing_loss.item())
                target_list.append(label)
                _, prediction = torch.max(prediction, 1)
                #print('label : ', label, ' prediction : ', prediction)
                pred_idx_list.append(prediction)
                total += 1
                correct += (prediction == label).sum().item()

            testing_loss = np.mean(testing_loss_list)
            test_accuracy = (100 * correct / total)

            pred_idx_numpy = torch.stack(pred_idx_list).cpu().numpy() # shape : [batch_size, 1]
            pred_idx_numpy = np.squeeze(pred_idx_numpy) # shape : [batch_size]
            target_numpy = torch.stack(target_list).cpu().numpy()  # shape : [batch_size]

            print('Accuracy of the model on the {} dataset : {:.4f}'.format(mode, test_accuracy))
            print('loss of the model on the {} dataset : {:.4f}'.format(mode, testing_loss))

            self.test_loss_tracker.append(testing_loss)
            self.test_acc_tracker.append(test_accuracy * 0.01)

        # Save model checkpoints.
        #if epoch >= self.save_start:
        if True:
            if test_accuracy > self.max_acc:
                #self.early_cnt = 0  # test_auc is improved early_cnt is reset to 0.
                self.max_acc = test_accuracy
                model_path = os.path.join(self.model_dir, '{}.ckpt'.format(epoch + 1))
                torch.save(self.model.state_dict(), model_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))
                self.save_flag = True
            """
            elif epoch >= self.early_start:
                print('self.early_cnt is increased')
                self.early_cnt += 1
                #plt.close()
            """

        if self.save_flag:
            self.save_flag = False
            self.plotingConfusionMatrix(target_numpy, pred_idx_numpy, self.label_list, epoch, self.sample_dir, prefix='test_')
            txt_path = osp.join(self.sample_dir, 'test_accuracy_epoch_{}.txt'.format(str(epoch).zfill(3)))
            with open(txt_path, 'w') as fp:
                print('acc : {}'.format(correct / float(total)), file=fp)

        if self.use_tensorboard:
            self.logger.scalar_summary('{} accuracy'.format(mode), test_accuracy, epoch)
            self.logger.scalar_summary('{}_loss'.format(mode), testing_loss, epoch)
            self.logger.param_summary(self.model, epoch)

        self.model = self.model.train()


