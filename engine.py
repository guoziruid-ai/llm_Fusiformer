import csv
import math
import os
import sys
from typing import Iterable

import torch

from tools import misc, lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    amp=True, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (inputs, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if isinstance(inputs, dict):
            for key, value in inputs.items():
                if isinstance(value, (list, tuple)):
                    inputs[key] = [x.to(device, non_blocking=True) for x in value]
                elif hasattr(value, 'to'):
                    inputs[key] = value.to(device, non_blocking=True)
        elif isinstance(inputs, (list, tuple)):
            inputs = [x.to(device, non_blocking=True) for x in inputs]
        else:
            inputs = inputs.to(device, non_blocking=True)

        if isinstance(targets, list):
            targets = torch.cat(targets, -1).to(device, non_blocking=True)
        else:
            targets = targets.to(device, non_blocking=True)

        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    loss = sum([criterion(output, targets) for output in outputs])
                else:
                    loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                loss = sum([criterion(output, targets) for output in outputs])
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=True)

        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True):
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        if isinstance(batch, tuple) and len(batch) == 2:
            batch_inputs, targets = batch
        else:
            batch_inputs, targets = batch, None
        
        # 关键修改：确保传递给模型的是字典格式
        if isinstance(batch_inputs, dict):
            # 已经是字典，直接使用
            inputs = batch_inputs
        elif isinstance(batch_inputs, list) and len(batch_inputs) == 2:
            # 如果是 [graph, line_graph] 格式，包装成字典
            inputs = {
                'graph_input': batch_inputs,  # [graph, line_graph]
                'text': None  # 如果没有文本，设为None
            }
        elif isinstance(batch_inputs, tuple) and len(batch_inputs) == 2:
            # 如果是 (graph_data, text_data) 格式
            inputs = {
                'graph_input': batch_inputs[0] if isinstance(batch_inputs[0], list) else batch_inputs[0],
                'text': batch_inputs[1] if len(batch_inputs) > 1 else None
            }
        else:
            # 其他情况，尝试包装
            inputs = {
                'graph_input': batch_inputs,
                'text': None
            }
        
        # 将inputs中的所有张量移动到device
        def to_device(data, device):
            if isinstance(data, dict):
                return {k: to_device(v, device) for k, v in data.items()}
            elif isinstance(data, list):
                return [to_device(x, device) for x in data]
            elif isinstance(data, tuple):
                return tuple(to_device(x, device) for x in data)
            elif hasattr(data, 'to'):
                return data.to(device, non_blocking=True)
            else:
                return data
        
        inputs = to_device(inputs, device)
        targets = targets.to(device, non_blocking=True)

        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        mse = torch.nn.functional.mse_loss(outputs, targets)
        mae = torch.nn.functional.l1_loss(outputs, targets)

        batch_size = targets.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['mae'].update(mae.item(), n=batch_size)
        metric_logger.meters['mse'].update(mse.item(), n=batch_size)
    
    metric_logger.synchronize_between_processes()
    print('* MAE {mae.global_avg:.3f} MSE {mse.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(mae=metric_logger.mae, mse=metric_logger.mse, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_evaluate(data_loader, model, device, amp=False, save_path="test_mae_new.csv"):
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    with open(save_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['sample_index', 'mse', 'mae'])

        sample_index = 0

        for inputs, targets in metric_logger.log_every(data_loader, 1, header):
            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device, non_blocking=True) for x in inputs]
            else:
                inputs = inputs.to(device, non_blocking=True)
            if isinstance(targets, list):
                targets = torch.cat(targets, -1).to(device, non_blocking=True)
            else:
                targets = targets.to(device, non_blocking=True)
            

            # compute output
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            mse = torch.nn.functional.mse_loss(outputs, targets)
            mae = torch.nn.functional.l1_loss(outputs, targets)



            metric_logger.update(loss=loss.item())
            metric_logger.meters['mae'].update(mae.item(), n=1)
            metric_logger.meters['mse'].update(mse.item(), n=1)
            csv_writer.writerow([sample_index, mse.item(), mae.item()])

            sample_index += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* MAE {mae.global_avg:.3f} MSE {mse.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(mae=metric_logger.mae, mse=metric_logger.mse, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



