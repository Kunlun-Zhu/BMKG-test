import torch
import torch.nn.functional as F
import math
import os
import bmtrain as bmt
import wandb
import numpy as np
import time


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Trainer():
    def __init__(self, model, train_config):
        self.model = model
        self.model_name = train_config["model_name"]

        self.do_train = train_config["do_train"]
        self.do_val = train_config["do_val"]
        self.do_test = train_config["do_test"]

        self.use_cuda = train_config["use_cuda"]
        self.gpu_ids = train_config["gpu_ids"]
        self.gpus = len(self.gpu_ids)
        self.node = train_config["node"]
        self.bmtrain = train_config["bmtrain"]
        if self.bmtrain:
            self.device = torch.device('cuda:{}'.format(bmt.rank()))
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]) if self.use_cuda else 'cpu')

        self.batch_size = train_config["batch_size"]
        self.epoch = train_config["epoch"]
        self.learning_rate = train_config["learning_rate"]
        self.skip_steps = train_config["skip_steps"]
        self.save_path = train_config["save_path"]
        self.checkpoint_num = train_config["checkpoint_num"]
        self.log_file = open(os.path.join(self.save_path, 'train.log'), 'w', encoding='utf-8')

        self.pad_id = train_config["pad_id"]

        self.loss_function = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.best_loss = 10000.0
        self.last_loss = 10000.0
        self.save = False

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def load_train_data_loader(self, train_data_loader):
        self.train_data_loader = train_data_loader

    def load_val_data_loader(self, val_data_loader):
        self.val_data_loader = val_data_loader

    def prediction(self):
        pass

    def validation(self, epoch):
        if self.use_ema:
            self.ema_model.apply_shadow()
        self.model.eval()
        acc, loss_sum = 0.0, 0.0
        step_per_epoch = len(self.val_data_loader)
        print("----------- validation -----------")
        self.log_file.write(str(len(self.val_data_loader)) + '\n')
        for iter, batch_data in enumerate(self.val_data_loader):
            # fetch batch data
            try:
                src_id, pos_id, input_mask, mask_pos, mask_label = batch_data
            except RuntimeError:
                print("Per data instance's length should be 5, received {}.".format(len(batch_data)))
                continue
            if self.use_cuda:
                src_id = src_id.to(self.device)
                src_id, pos_id, input_mask, mask_pos, mask_label = \
                    src_id.to(self.device), pos_id.to(self.device), input_mask.to(self.device), mask_pos.to(
                        self.device), mask_label.to(self.device)
            input_x = {
                'src_ids': src_id,
                'position_ids': pos_id,
                'input_mask': input_mask,
                'mask_pos': mask_pos,
                'mask_label': mask_label
            }
            torch.cuda.synchronize()
            # forward
            logits = self.model(input_x)["logits"]
            # loss
            label = mask_label
            loss = self.loss_function(
                input=logits,
                target=mask_label.squeeze()
            )

            if self.bmtrain:
                global_loss = bmt.sum_loss(loss).item()
                loss = self.optimizer.loss_scale(loss)
                loss_sum += global_loss
            else:
                loss_sum += loss.data
            acc += logits.max(dim=1)[1].eq(mask_label.squeeze()).sum().data

        # acc = acc * 100 / float(len(self.val_data_loader.dataset))
        acc = acc * 100 / 100000.0
        loss_sum = loss_sum / len(self.val_data_loader)
        if not self.bmtrain:
            loss_sum = loss_sum.cpu().detach().numpy()
        if self.bmtrain:
            bmt.print_rank('loss:{}, acc:{}.'.format(str(loss_sum), str(acc)))
            bmt.synchronize()
        else:
            print('loss:{}, acc:{}.'.format(str(loss_sum), str(acc)))
        # wandb.log({"validation loss": loss_sum})
        self.log_file.write(('loss:{}, acc:{}%.'.format(str(loss_sum), str(acc))) + '\n')

        # save best checkpoint
        if self.last_loss <= loss_sum:
            self.save = True
        self.last_loss = loss_sum
        if self.best_loss > loss_sum and self.save:
            self.best_loss = loss_sum
            if self.use_ema:
                self.ema_model.apply_shadow()
            if isinstance(self.model, torch.nn.DataParallel):
                torch.save(self.model.module.state_dict(), self.save_path + '{}_lr{}_bs{}_ema{}_epoch{}.pt'.format(
                    self.model_name, self.learning_rate,
                    self.batch_size, str(self.use_ema), str(epoch)))
            elif self.bmtrain:
                bmt.save(self.model, self.save_path + '{}_bmt_lr{}_bs{}_ema{}_epoch{}.pt'.format(
                    self.model_name, self.learning_rate,
                    self.batch_size, str(self.use_ema), str(epoch)))
            else:
                torch.save(self.model.state_dict(), self.save_path + '{}_lr{}_bs{}_ema{}_epoch{}.pt'.format(
                    self.model_name, self.learning_rate,
                    self.batch_size, str(self.use_ema), str(epoch)))
            if self.use_ema:
                self.ema_model.restore()

        if self.use_ema:
            self.ema_model.restore()

    def train(self):
        self.model.train()
        acc, loss_sum = 0.0, 0.0
        step_per_epoch = len(self.train_data_loader)
        total_train_step = step_per_epoch * self.epoch
        save_step = math.ceil(total_train_step / self.node / self.gpus / self.checkpoint_num)
        for epoch in range(self.epoch):
            start_time = time.time()
            # label_smoothing = 0.0 if epoch < 5 else 0.0
            for iter, batch_data in enumerate(self.train_data_loader):
                # fetch batch data
                h_desc, r_desc, t_desc = batch_data
                batch = dict()
                print(r_desc)
                h_tokenized_inputs = self.model.tokenizer(h_desc, return_tensors="pt")
                # r_tokenized_inputs = self.model.tokenizer(r_desc, return_tensors="pt")
                t_tokenized_inputs = self.model.tokenizer(t_desc, return_tensors="pt")
                batch['mlm_inputs'] = h_tokenized_inputs
                # print(h_tokenized_inputs['input_ids'])
                batch['mlm_targets'] = torch.zeros(size=[1], dtype=torch.int64)
                batch['ke_targets'] = t_tokenized_inputs['input_ids']
                print(t_tokenized_inputs['input_ids'])
                batch['classification_head'] = False

                batch['ke_inputs'] = dict()
                batch['ke_inputs']["heads"] = h_tokenized_inputs['input_ids']
                batch['ke_inputs']["tails"] = t_tokenized_inputs['input_ids']
                batch['ke_inputs']["nheads"] = h_tokenized_inputs['input_ids']
                batch['ke_inputs']["ntails"] = t_tokenized_inputs['input_ids']
                batch['ke_inputs']["heads_r"] = h_tokenized_inputs['input_ids']
                batch['ke_inputs']["tails_r"] = t_tokenized_inputs['input_ids']

                loss = self.model.train_step(batch)

                # backward
                if self.bmtrain:
                    global_loss = bmt.sum_loss(loss).item()
                    loss = self.optimizer.loss_scale(loss)
                    loss.backward()
                    bmt.optim_step(self.optimizer)
                    lr = self.lr_scheduler.get_lr()
                    torch.cuda.synchronize()
                else:
                    loss.mean().backward()
                    self.optimizer.step()
                    lr = self.lr_scheduler.get_last_lr()[0]

                # log

                # print(logits.max(dim=1)[1])
                current_step = epoch * step_per_epoch + iter
                print('Epoch:{}, Step:{}/{}, loss:{}, lr:{}.'.format(
                    str(epoch), str(current_step),
                    str(total_train_step),
                    str(loss.cpu().detach().numpy() if not self.bmtrain else global_loss),
                    str(lr)
                )
                )
                if iter % (step_per_epoch / 2) == 0 or iter + 1 == step_per_epoch:
                    if not self.bmtrain:
                        print('Epoch:{}, Step:{}/{}, loss:{}, lr:{}.'.format(
                            str(epoch), str(current_step),
                            str(total_train_step),
                            str(loss.cpu().detach().numpy() if not self.bmtrain else global_loss),
                            str(lr)
                            )
                        )
                    else:
                        bmt.print_rank('Epoch:{}, Step:{}/{}, loss:{}, lr:{}.'.format(
                            str(epoch), str(current_step),
                            str(total_train_step),
                            str(loss.cpu().detach().numpy() if not self.bmtrain else global_loss),
                            str(lr)
                            )
                        )

                        # bmt.synchronize()
                    # wandb.log({
                    #     "training loss": loss.cpu().detach().numpy(),
                    # })

                    self.log_file.write('Epoch:{}, Step:{}/{}, loss:{}, lr:{}.'.format(
                        str(epoch), str(current_step),
                        str(total_train_step),
                        str(loss.cpu().detach().numpy() if not self.bmtrain else global_loss),
                        str(lr)) + '\n'
                    )
                    self.log_file.flush()
            # wandb.log({
            #     "learning rate": np.array(float(lr))
            # })
            # torch.cuda.synchronize()
            # self.validation(epoch)
            epoch_time = time.time() - start_time
            if self.bmtrain:
                bmt.print_rank("time:" + str(epoch_time))
            else:
                print("time:" + str(epoch_time))
            self.model.train()
            self.lr_scheduler.step() if not self.bmtrain else bmt.optim_step(self.lr_scheduler)

        return acc, loss_sum
