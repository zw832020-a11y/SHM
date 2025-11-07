_base_ = 'grounding_dino_swin-l_pretrain_all.py'



dataset_type = 'CocoDataset'
data_root = 'data/'  # 请将此处修改为你 COCO 数据集存放的根目录

# 获取基础配置中的测试流水线，并设置需要保留的元信息
base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')

class_name = (
    'person',
    'ear',
    'ear-mufs',
    'face',
    'face-guard',
    'face-mask',
    'foot',
    'tool',
    'glasses',
    'gloves',
    'helmet',
    'hands',
    'head',
    'medical-suit',
    'shoes',
    'safety-suit',
    'safety-vest'
)
metainfo = dict(classes=class_name)

# 定义测试数据集（这里以 COCO 2017 验证集为例）
test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='SH17_val_output_coco.json',  # COCO 标注文件
    data_prefix=dict(img='val/'),    # 图片所在子目录
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True,
    metainfo=metainfo  # 添加metainfo配置
)

# 设置数据加载器，每次一个样本（可根据需要调整 batch_size 和 num_workers）
test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=False
)

# 使用 COCO Metric 评估检测框的表现（mAP等指标）
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'SH17_val_output_coco.json',
    metric='bbox'
)

# # 设置可视化配置
# default_hooks = dict(
#     visualization=dict(
#         type='DetVisualizationHook',
#         draw=True,  # 在命令行参数中使用--show或--show_dir时会自动设置为True
#         interval=1,
#         score_thr=0.3
#     )
# )

