import torch
import os
import torch.nn as nn
import numpy as np
import rclpy
import torch.optim as optim
from tqdm import tqdm
from RadarDetection.data.data_loader import DataLoaderSelf
from RadarDetection.runners.base_runner import BaseRunner
from RadarDetection.utils.plot_loss import plot_rewards
from collections import deque
Subscriber = None


class Runner(BaseRunner):
    def __init__(self, configs):
        super(Runner, self).__init__(configs)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.frame = None
        self.history_frame = None
        self.history_frame_list = None
        self.frame_buffer = None
        self.subscriber = None
        self.RADAR_ON = self.config.RADAR_ON
        self.combine_mode = self.config.combine_mode
        self.subscriber_on = False
        if self.RADAR_ON:
            self.subscriber_on = True
        if self.config.Train:
            self.tq_bar = tqdm(range(self.epochs))
        else:
            self.tq_bar = None
        self.history_frame_len = self.config.history_frame_len
        self.start = True

        self.mini_loss = np.inf
        self.loss_mean = np.inf
        self.init_loss = np.inf

        # post add
        self.loss_record = []

    def OnCreate(self, model):
        self.GetDataSet()
        self.RadarSetInfo()
        self.model = model(self.configs).to(self.device)
        self.model.OnCreate()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.MakeCriterion()
        self.SetModelPath()

    def Execute(self):
        if self.RADAR_ON:
            self.RadarSubscriberSpin()
        elif self.config.Train:
            self.Train()
        elif self.combine_mode:
            self.CombineEval()
        else:
            self.Eval()

    def Train(self):
        if self.continue_train:
            try:
                self.model.load_state_dict(torch.load(self.model_load_path))
            except Exception as e:
                print()
                print(f"Model Load Path Error: {e}")
        start_flag = True
        for epoch in self.tq_bar:
            for batch_idx, (data, target) in enumerate(self.data_loader):
                self.tq_bar.set_description(f"Epoch [{epoch + 1}/{self.epochs}]")
                if data is False:
                    print("data valid! continue")
                    continue
                data, target = data.to(self.device), target.to(self.device)

                # forward
                self.RadarGetData(frame=data, history_frame_len=self.history_frame_len)
                outputs = self.model(self.frame, self.history_frame)

                # loss cal
                loss = self.criterion(outputs, target)

                # backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print info
                self.loss_record.append(loss.item())
                if len(self.loss_record) > 10:
                    self.loss_mean = np.mean(self.loss_record[-10:])
                elif len(self.loss_record) == 10 and epoch == 0:
                    self.init_loss = np.mean(self.loss_record[-10:])
                # Init_reward = ep_mean_reward if episode == 10 else 0
                # bf_reward = episode_reward if episode_reward <= bf_reward else bf_reward
                if self.loss_mean <= self.mini_loss:
                    self.mini_loss = self.loss_mean
                    if epoch >= 10:
                        torch.save(self.model.state_dict(), self.model_save_path)
                now_step_with_steps = batch_idx + 1 / len(self.data_loader)
                self.tq_bar.set_postfix({'STEP': f'{now_step_with_steps:.0f}',
                                         'INIT LOSS': f'{self.init_loss:.5f}',
                                         'BEST': f'{self.mini_loss:.5f}',
                                         'LAST LOSS': f'{self.loss_mean:.5f}'})
            if start_flag:
                print()
                start_flag = False
            if epoch >= self.epochs * 0.25 and epoch % (self.epochs // 4) == 0:
                print()
            if epoch >= self.epochs * 0.5 and epoch % (self.epochs // 2) == 0:
                print()
            if epoch >= self.epochs * 0.75 and epoch % (self.epochs * 3 // 4) == 0:
                print()

                # if (batch_idx+1) % 1 == 0:
                # print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        print(" ---------------------------- Finished Training ---------------------------- ")
        print("-- Post Processing...")
        print()
        self.PostProcess()
        print()
        print("-- Done! Succeed.")

    def Eval(self):
        print(" ---------------------------- Eval Mode ---------------------------- ")
        try:
            print("-- Load model")
            self.model.load_state_dict(torch.load(self.model_load_path))
            print("-- Load model succeed!")
        except Exception as e:
            print(f"Model Load Path Error: {e}")
        self.model.eval()
        print("-- Set model to eval")
        print()

        for i in range(self.config.detect_num):
            data, target = self.dataset[0]
            data = data.unsqueeze(0).to(self.device)
            target = target.unsqueeze(0).to(self.device)

            history_frame = data
            outputs = self.model(data, history_frame)
            loss = self.criterion(outputs, target)

            # print
            outputs = outputs[0].tolist()
            target = target[0].tolist()
            for i in range(int(len(outputs)/2)):
                print(" ------ output {:2}: ({:6.2f}, {:6.2f}), target {:2}: ({:6.2f}, {:6.2f})".
                      format(i+1, outputs[i*2], outputs[(i+1)*2-1], i+1, target[i*2], target[(i+1)*2-1]))
            print(" ---- loss = ", loss.item())
            print()
        print("-- Done! Succeed.")

    def CombineEval(self):
        if self.start:
            print(" ---------------------------- Get Mode ---------------------------- ")
            try:
                print("-- Load model")
                print("-- (Model path:", self.model_load_path, ")")
                self.model.load_state_dict(torch.load(self.model_load_path))
                print("-- Load model succeed!")
            except Exception as e:
                print(f"Model Load Path Error: {e}")
            self.model.eval()
            print("                                             -- Init Model Succeed!")
            print()
            self.start = False

        self.config.RADAR = []
        data, target = self.dataset[0]
        data = data.unsqueeze(0).to(self.device)
        target = target.unsqueeze(0).to(self.device)

        history_frame = data
        outputs = self.model(data, history_frame)

        # print
        outputs = outputs[0].tolist()
        target = target[0].tolist()
        for i in range(self.config.target_num):
            self.config.RADAR.append(outputs[2 * i: 2 * i + 2])
            self.config.RADAR.append(target[2 * i: 2 * i + 2])

        if self.config.end:
            print("-- Reached the end! Succeed.")

    def EvalSingle(self):
        print(" ---------------------------- Eval Mode ---------------------------- ")
        try:
            print("-- Load model")
            self.model.load_state_dict(torch.load(self.model_load_path))
            print("-- Load model succeed!")
        except Exception as e:
            print(f"Model Load Path Error: {e}")
        self.model.eval()
        print("-- Set model to eval")
        print()

        data, target = self.dataset[1]
        data = data.unsqueeze(0).to(self.device)
        target = target.unsqueeze(0).to(self.device)

        history_frame = data
        outputs = self.model(data, history_frame)
        loss = self.criterion(outputs, target)

        # print
        outputs = outputs[0].tolist()
        target = target[0].tolist()
        for i in range(int(len(outputs)/2)):
            print(" ------ output {:2}: ({:6.2f}, {:6.2f}), target {:2}: ({:6.2f}, {:6.2f})".
                  format(i+1, outputs[i*2], outputs[(i+1)*2-1], i+1, target[i*2], target[(i+1)*2-1]))
        print(" ---- loss = ", loss.item())
        print()
        print("-- Done! Succeed.")

    def RadarTimerCallBack(self, frame):
        """
        :param frame: Real time acquisition of current frame echo image from radar
        :param history_frame: The first [history_frame_len] echo images of the current frame
        """

        self.RadarGetData(frame, history_frame_len=1)
        self.RadarProcessing()

    def RadarGetData(self, frame, history_frame_len=2):
        """
        :param frame: store current frame
        :param history_frame_len: get and store history frame of len=history_frame_len
        """
        if history_frame_len > self.config.frame_buffer_capacity:
            raise ValueError("The frame_buffer_capacity should be greater then history_frame_len")

        self.frame = frame
        if len(self.frame_buffer) == 0:
            self.history_frame = frame
            for i in range(history_frame_len-1):
                self.history_frame = torch.cat((self.history_frame, frame), dim=1)
        elif len(self.frame_buffer) >= history_frame_len:
            self.history_frame = torch.cat(list(self.frame_buffer)[-history_frame_len:], dim=1)
        else:
            self.history_frame = torch.cat(list(self.frame_buffer)[:], dim=1)
            for i in range(history_frame_len - len(self.frame_buffer)):
                self.history_frame = torch.cat((self.history_frame, frame), dim=1)

        self.frame_buffer.append(frame)

    def RadarSetInfo(self, device=None):
        if device is not None:
            self.device = torch.device(device)
        self.frame_buffer = deque(maxlen=self.config.frame_buffer_capacity)
        self.history_frame = None

    def RadarLoadModel(self):
        if not self.run_dir.exists():
            raise ValueError(f"There does not exist path: {self.run_dir}")

        if not os.path.exists(self.model_load_path):
            raise ValueError(f"Model path does not exist: {self.model_load_path}")
        else:
            self.model.load_state_dict(torch.load(self.model_load_path))

    def RadarSubscriberSpin(self, args=None):
        if not self.RADAR_ON:
            raise ValueError("Error! Not using radar!")
        elif not self.subscriber_on:
            raise ValueError("Error! Subscriber can not be created!")
        elif Subscriber is None:
            raise ValueError("Error! Subscriber can not be created!")
        else:
            print(" ----- Radar On!")

        try:
            self.RadarSetInfo("cpu")
            self.RadarLoadModel()
        except Exception as e:
            print(f" Radar Init Failed!: {e}")

        rclpy.init(args=args)
        self.subscriber = Subscriber(self)
        rclpy.spin(self.subscriber)
        self.subscriber.destroy_node()
        rclpy.shutdown()

    def RadarProcessing(self):
        """
        here input radar [current frame, history_frame] to the model,
        and then output the process result, i.e, target detection result: (distance, direction)
        :input: frame, history_frame
        :output: distance, direction
        :return: detection results -> outputs, to the next step which is radar target tracking
        """
        print(" ---------------------------- Radar Mode ---------------------------- ")

        outputs = self.model(self.frame, self.history_frame)
        outputs = outputs[0].tolist()
        return outputs

    def CommonSetInfo(self):
        pass

    def GetDataSet(self):
        self.dataset, self.data_loader = DataLoaderSelf(self.config)

    def MakeCriterion(self, ctype=None):
        if ctype is None and self.ctype is not None:
            ctype = self.ctype

        if ctype == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()
        elif ctype == "MSE":
            self.criterion = nn.MSELoss()
        elif ctype == "NLL":
            self.criterion = nn.NLLLoss()
        elif ctype == "BCE":
            self.criterion = nn.BCELoss()
        elif ctype == "L1":
            self.criterion = nn.L1Loss()
        elif ctype == "SmoothL1":
            self.criterion = nn.SmoothL1Loss()
        elif ctype == "MultiLabelSoftMargin":
            self.criterion = nn.MultiLabelSoftMarginLoss()
        elif ctype == "MultiMargin":
            self.criterion = nn.MultiMarginLoss()
        else:
            raise ValueError('Not assign criterion. '
                             'Please assign a criterion type.')

    def Criterion(self, outputs, target):
        self.criterion = None

    def SetModelPath(self, model_name=None):
        if not self.run_dir.exists():
            raise ValueError(f"There does not exist path: {self.run_dir}")

        if model_name is None:
            self.model_save_path = self.run_dir / self.save_name
            self.model_load_path = self.model_save_path
        else:
            self.model_save_path = self.run_dir / model_name
            self.model_load_path = self.model_save_path

    def SaveModel(self):
        torch.save(self.model.state_dict(), self.model_save_path)

    def PostProcess(self):
        plot_dir = self.run_dir.parent.parent.parent
        save_path = plot_dir / 'LossResult/reward.pdf'
        save_path_png = plot_dir / 'LossResult/reward.png'
        print(" ---- plot directory = ", plot_dir)
        plot_rewards(self.loss_record, save_path, save_path_png)
        print(" ---- plot save path = ", save_path)
