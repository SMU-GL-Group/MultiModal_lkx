#coding=utf-8
"""
各种预加载函数。
"""
import SimpleITK as sitk
import logging
import pydicom
import yaml
from easydict import EasyDict
import os
import glob
import torch


# -------- 加载DICOM数据 --------
def load_dcm_data(folder_path):
    """
    载入DICOM数据。
    :param folder_path: 患者DICOM数据路径。
    :return: image_data: ndarray数组，metadata: 包含元数据的字典。
    """
    logging.info(f'正在加载DICOM数据: {folder_path}')

    try:
        # 读取DICOM文件
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path)

        if not dicom_names:
            logging.warning(f'未找到DICOM文件: {folder_path}')
            return None, None

        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        image_data = sitk.GetArrayFromImage(image)

        # 提取元数据
        metadata = extract_metadata(dicom_names[0])

        logging.info(f'成功加载DICOM数据: {len(dicom_names)}个文件')
        return image_data, metadata

    except Exception as e:
        logging.error(f'加载DICOM数据失败: {str(e)}')
        return None, None


def extract_metadata(dicom_file):
    """从DICOM文件中提取元数据"""
    try:
        dcm_data = pydicom.dcmread(dicom_file)
        metadata = {
            'PatientName': dcm_data.get('PatientName', 'N/A'),
            'PatientID': dcm_data.get('PatientID', 'N/A'),
            'PatientBirthDate': dcm_data.get('PatientBirthDate', 'N/A'),
            'PatientSex': dcm_data.get('PatientSex', 'N/A'),
            'StudyDate': dcm_data.get('StudyDate', 'N/A'),
            'StudyTime': dcm_data.get('StudyTime', 'N/A'),
            'Modality': dcm_data.get('Modality', 'N/A'),
            'InstitutionName': dcm_data.get('InstitutionName', 'N/A'),
            'Manufacturer': dcm_data.get('Manufacturer', 'N/A'),
            'ManufacturerModelName': dcm_data.get('ManufacturerModelName', 'N/A'),
            'DeviceSerialNumber': dcm_data.get('DeviceSerialNumber', 'N/A'),
            'SoftwareVersions': dcm_data.get('SoftwareVersions', 'N/A'),
            'ProtocolName': dcm_data.get('ProtocolName', 'N/A'),
            'StudyDescription': dcm_data.get('StudyDescription', 'N/A'),
            'SeriesDescription': dcm_data.get('SeriesDescription', 'N/A'),
            'FilePath': dicom_file,
            'RescaleSlope': dcm_data.get('RescaleSlope', 1.0),
            'RescaleIntercept': dcm_data.get('RescaleIntercept', 0.0),
            'WindowCenter': dcm_data.get('WindowCenter', 40.0),
            'WindowWidth': dcm_data.get('WindowWidth', 400.0),
        }
        return metadata
    except Exception as e:
        logging.error(f'提取元数据失败: {str(e)}')
        return {}


# -------- 加载配置文件 --------
def load_config(config_path):
    """
    载入配置文件，返回字典形式。
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


# -------- 检查并载入checkpoint --------
def load_checkpoint(checkpoint_dir, model, optimizer, scheduler, device, k):
    """
    Args:
        checkpoint_dir: 保存点的目录。
        model: 要训练的模型。
        optimizer: 训练用的优化器。
        scheduler: 学习率调度其。
        device: 要加载到的显卡。
        k: 第k折。
    Returns: 当前训练开始的轮次，总的迭代步数。
    """
    start_epoch, global_step = 0, 0
    # 检查目录是否存在
    if not os.path.exists(checkpoint_dir):
        logging.info(f"\nCheckpoint directory '{checkpoint_dir}' does not exist yet.")
        return start_epoch, global_step

    # 查找最新的检查点文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f'last_model_k={k}.pth'))
    if not checkpoint_files:
        logging.info(f"No checkpoint files found in '{checkpoint_dir}'. Starting from epoch 0.")
        return start_epoch, global_step

    # 加载最新的检查点文件
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint, map_location=device)

    # 恢复模型和优化器的状态，以及总迭代步数
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_dict'])
    start_epoch = checkpoint['epoch']  # 从当前epoch重新开始训练
    # global_step = checkpoint['global_step']

    logging.info(f'Checkpoint loaded from {latest_checkpoint} at epoch {start_epoch}')
    logging.info(f'Training from epoch {start_epoch}')  # 将信息记录到日志文件中
    return start_epoch, global_step
