#!/usr/bin/env python3
"""Generate the versioned 500-record smart-ward NLU dataset.

The raw records follow docs/20-NLU数据集构建说明-P5.md.  The script also
derives Qwen ChatML views and cloud inference request fixtures without changing
the canonical labels.  Generation is deterministic so every artifact can be
reproduced from source.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


VERSION = "1.0.0"
SEED = 20260809
SYSTEM_PROMPT = (
    "你是智慧病房护理助手。根据事件快速输出：状况、紧急程度（紧急/警告/提醒）、"
    "最多3条护理建议。只做护理建议，不做诊断。"
)

COUNTS = {
    "fall_suspected": 50,
    "seizure": 50,
    "fall_prediction": 50,
    "nurse_call": 40,
    "bed_leave": 40,
    "abnormal_posture": 40,
    "long_still": 40,
    "night_wandering": 35,
    "door_departure": 35,
    "bedsore_risk": 35,
    "environment_anomaly": 30,
    "device_fault": 30,
    "node_offline": 25,
}

URGENCY = {"P1": "紧急", "P2": "警告", "P3": "提醒"}

# Each judgment has four training scenario families and one held-out test family.
# Test descriptions therefore do not reuse a scenario family from training.
EVENTS: Dict[str, Dict[str, Any]] = {
    "fall_suspected": {
        "cn": "疑似跌倒", "priority": "P1", "hours": [0, 6, 9, 14, 19, 23],
        "rules": ["camera.posture=falling", "fall_score>0.5", "bed_sensor.on_bed=false"],
        "scenes": {
            "confirm": ["患者从床沿滑落到地面，姿态由坐位快速转为横卧", "患者行走时突然失去平衡倒地，倒地后未立即起身", "床旁检测到快速下坠动作，随后持续侧卧于地面", "患者起身时向前扑倒，床垫同时报告离床", "夜间患者从床边跌落，地面区域持续检测到人体"],
            "reject": ["患者弯腰拾取拖鞋后自行站直，床垫始终显示在床", "护理人员协助患者由床转移到轮椅，动作平稳", "被褥滑落遮挡镜头形成横向轮廓，画面中未见人体倒地", "摄像头短暂抖动造成检测框下移，患者姿态未改变", "患者在床上正常翻身，床垫压力与生命体征均稳定"],
            "escalate": ["患者快速下蹲后离开画面，地面区域被床栏遮挡", "摄像头显示疑似倒地，但床垫仍报告在床", "仅检测到上半身向床外倾斜，后续画面暂时丢失", "患者滑坐至床边，无法确定是否接触地面", "低照度下检测到横卧轮廓，关键点置信度不足"],
        },
        "advice": {
            "confirm": ["立即到场评估意识和外伤，保持原位并通知医生", "立即查看患者，检查呼吸意识，按院内跌倒流程处置", "先确认现场安全并评估伤情，必要时启动急救响应"],
            "reject": ["人工复核画面与床垫状态，确认安全后继续观察", "核对患者当前体位，校准摄像头并保留复核记录", "确认患者无异常后关闭误报，继续常规巡视"],
            "escalate": ["立即到床旁复核，补查地面盲区并持续观察", "优先人工确认患者位置，必要时按跌倒事件处理", "调取连续画面并检查患者状态，无法排除时升级处置"],
        },
    },
    "seizure": {
        "cn": "抽搐检测", "priority": "P1", "hours": [1, 5, 10, 15, 20, 23],
        "rules": ["camera.tremor_score>0.7", "repetitive_motion=true"],
        "scenes": {
            "confirm": ["患者四肢出现持续节律性抖动，呼叫后无即时回应", "患者躯干僵直并伴双臂反复抽动，持续超过二十秒", "床上检测到高频重复动作，头部与上肢同步抖动", "患者突然倒卧并出现连续肢体震颤", "夜间患者全身反复抽动，床垫压力呈周期性变化"],
            "reject": ["患者清醒状态下主动进行手臂康复训练，动作按固定节奏完成", "患者因寒冷短暂发抖，加被后动作立即停止", "护理人员整理床铺造成被褥周期晃动，人体关键点稳定", "患者正常翻身并伸展四肢，动作结束后继续休息", "摄像头画面压缩抖动，床垫和人体姿态均无异常"],
            "escalate": ["单侧手臂间歇抖动，但面部区域被遮挡", "患者出现短促重复动作，持续时间不足且呼叫反应未知", "画面显示腿部震颤，床垫信号未同步", "高频动作只出现数帧，随后摄像头失去目标", "患者睡眠中有不规则抖动，无法区分梦动与异常发作"],
        },
        "advice": {
            "confirm": ["立即到场保护头部并保持呼吸道通畅，记录时长并呼叫医生", "移开周围硬物，勿强行按压肢体，立即通知医生", "立即查看患者并记录发作表现，按院内急症流程响应"],
            "reject": ["人工确认患者清醒且动作可控，记录误报并继续观察", "核对连续画面与床垫信号，确认安全后恢复常规监测", "确认无持续异常动作后关闭告警，并检查设备稳定性"],
            "escalate": ["立即床旁复核并持续计时，无法排除时通知医生", "观察意识和动作变化，补充连续证据后升级研判", "优先确认患者反应，异常持续或加重时按急症处置"],
        },
    },
    "fall_prediction": {
        "cn": "坠床预警", "priority": "P1", "hours": [2, 7, 11, 16, 21, 23],
        "rules": ["bed_sensor.on_bed=true", "posture=lying_edge", "fall_score>threshold"],
        "scenes": {
            "confirm": ["患者持续侧卧在床沿，半侧躯干已越过床垫边界", "患者坐在床沿向外探身，双脚尚未稳定着地", "睡眠中患者逐渐移向床边，肩部已贴近护栏缺口", "患者反复尝试自行下床，重心明显偏向床外", "患者翻身后停留在无护栏床沿，身体继续向外移动"],
            "reject": ["患者位于床中央，检测框因被褥褶皱误判为靠近床沿", "护理人员在旁协助患者安全坐起，床栏处于升起状态", "镜头角度变化导致床沿区域偏移，患者实际位置稳定", "患者短暂伸手取水后回到床中央", "空床上的枕头被识别成人体，床垫未检测到占床"],
            "escalate": ["患者靠近床沿但床栏遮挡了下肢位置", "床垫显示重心外移，摄像头关键点置信度偏低", "患者半坐起并向外倾斜，是否有人协助无法确认", "低照度下人体边界与床沿重叠，移动方向不明确", "患者在床沿停留较久，但后续姿态数据间歇丢失"],
        },
        "advice": {
            "confirm": ["立即协助患者回到安全体位，检查床栏并加强巡视", "立即到场防止坠床，调整体位并排查下床需求", "先保护患者避免继续外移，确认床栏和呼叫器可用"],
            "reject": ["复核床位区域和床垫信号，确认安全后继续观察", "校准床沿区域并检查床栏，无风险后记录误报", "确认患者位置稳定，修正检测区域后恢复监测"],
            "escalate": ["尽快到床旁确认体位，无法排除时先采取防坠措施", "补查床垫与连续画面，持续靠近床沿则立即干预", "优先人工复核患者位置，并保持呼叫器触手可及"],
        },
    },
    "nurse_call": {
        "cn": "护士呼叫", "priority": "P1", "hours": [0, 5, 9, 13, 18, 22],
        "rules": ["camera.call_requested=true"],
        "scenes": {
            "confirm": ["患者连续按下床旁呼叫按钮并抬手示意", "呼叫器产生有效按键信号，患者在床上等待响应", "患者通过床头终端主动发起护理请求", "呼叫按钮被按下两次，床旁检测到患者清醒", "夜间患者按铃并指向输液装置"],
            "reject": ["护理人员测试呼叫器线路，测试标记已同步上报", "清洁时误触呼叫按钮，患者表示没有护理需求", "按钮卡滞产生重复信号，患者不在床位", "设备重启回放了历史呼叫记录，时间戳不一致", "邻床呼叫信号串入当前节点，当前床位未按铃"],
            "escalate": ["收到一次短促呼叫信号，但床旁音频与画面均不可用", "呼叫器上报按键事件后立即离线", "患者疑似抬手示意，但按钮状态未成功同步", "连续出现弱信号，无法确定是按键还是线路抖动", "夜间检测到呼叫灯闪烁，但床位画面被遮挡"],
        },
        "advice": {
            "confirm": ["立即响应呼叫并询问需求，评估是否需要进一步协助", "尽快到床旁确认患者诉求，并记录响应结果", "立即联系患者并到场处理，必要时通知责任护士"],
            "reject": ["核对呼叫器记录与患者状态，确认误触后关闭告警", "检查按键和线路，确认患者无需求后记录设备问题", "人工确认床位无呼叫需求，并修复异常信号来源"],
            "escalate": ["优先联系并查看患者，同时检查呼叫设备状态", "尽快到床旁核实，无法联系时按有效呼叫处理", "人工复核呼叫来源，确认患者安全后排查线路"],
        },
    },
    "bed_leave": {
        "cn": "离床", "priority": "P2", "hours": [6, 8, 12, 15, 19, 22],
        "rules": ["bed_sensor.absence_seconds>=30", "camera.person_outside_bed=true"],
        "scenes": {
            "confirm": ["床垫连续四十五秒未检测到压力，患者出现在床旁区域", "患者自行下床后离开床位，离床时间已超过阈值", "床位压力消失且摄像头检测到患者向门口移动", "患者从床上起身，超过一分钟仍未返回", "夜间床垫报告离床，患者位于卫生间方向"],
            "reject": ["患者仍在床上，仅因侧卧使压力传感器短暂失去信号", "护理人员更换床单导致床垫暂时无压力", "床垫接头松动报告离床，但摄像头显示患者在床", "患者短暂坐到床边后立即返回，未超过离床阈值", "空床测试期间产生离床事件，床位当前未分配患者"],
            "escalate": ["床垫报告离床四十秒，但摄像头画面被遮挡", "摄像头检测到床旁人体，无法判断是患者还是护理人员", "床位压力间歇消失，患者下肢不在画面内", "患者可能已离床，但门区传感器未上报", "夜间压力信号消失，摄像头仅看到被褥移动"],
        },
        "advice": {
            "confirm": ["尽快确认患者去向和步态安全，必要时提供下床协助", "到床旁核实离床原因，检查是否需要如厕或活动协助", "确认患者位置并评估跌倒风险，提醒使用呼叫器"],
            "reject": ["核对床垫与画面，确认在床后检查传感器接触", "确认患者安全并校准离床阈值，记录本次误报", "检查床垫连接和床位状态，无异常后继续监测"],
            "escalate": ["尽快床旁确认患者位置，并补查门区与床垫记录", "人工复核遮挡区域，无法确认时按离床事件处理", "持续观察并联系患者，确认安全后再解除告警"],
        },
    },
    "abnormal_posture": {
        "cn": "异常体态", "priority": "P2", "hours": [1, 7, 10, 14, 18, 22],
        "rules": ["posture in curled,leaning,grabbing_chest"],
        "scenes": {
            "confirm": ["患者持续蜷缩并按压胸前，呼叫后动作未缓解", "患者坐位明显向一侧倾斜，无法自行恢复端正", "患者突然弯腰抱腹并保持异常姿势", "患者站立时身体持续前倾并抓住床栏支撑", "夜间患者蜷缩在床边，伴反复翻动"],
            "reject": ["患者弯腰整理鞋袜后恢复正常坐姿", "患者侧身取床头物品，短暂倾斜后自行复位", "康复训练中的指定动作被识别为异常体态", "被褥遮挡造成躯干关键点偏移，患者实际平卧", "护理人员协助翻身时出现短暂蜷缩姿态"],
            "escalate": ["患者间歇性按压胸前，但面部表情不可见", "躯干轻度偏斜持续十秒，随后关键点丢失", "患者蜷缩姿势与正常侧卧难以区分", "摄像头显示身体前倾，床垫状态没有变化", "低照度下检测到抓胸动作，但手部关键点置信度较低"],
        },
        "advice": {
            "confirm": ["尽快到场询问不适并评估意识和生命体征，必要时通知医生", "协助患者保持安全体位，观察症状并联系责任护士", "立即床旁复核异常姿态，持续不适时升级医疗评估"],
            "reject": ["确认患者动作自主且已恢复，记录误报并继续观察", "复核连续姿态与患者反馈，无异常后恢复常规监测", "检查关键点遮挡和镜头角度，确认安全后关闭告警"],
            "escalate": ["尽快询问患者感受并复测姿态，持续异常则通知医生", "床旁复核意识和生命体征，补充证据后决定是否升级", "保持观察并确认有无疼痛不适，无法排除时升级处置"],
        },
    },
    "long_still": {
        "cn": "长时间静止", "priority": "P2", "hours": [0, 4, 9, 13, 18, 23],
        "rules": ["position_duration>=300", "active_person=true"],
        "scenes": {
            "confirm": ["患者保持同一体位超过十分钟，呼叫后未见明显动作", "患者坐位静止时间显著超阈值，头部持续下垂", "患者卧床长时间无翻身，床垫压力分布未变化", "患者在床边保持不动，连续画面未检测到自主活动", "夜间患者异常静止且呼叫后没有姿态变化"],
            "reject": ["患者处于正常睡眠，呼吸微动和床垫信号稳定", "摄像头冻结导致画面不变，实时床垫数据仍有活动", "患者安静阅读，手部存在小幅连续动作", "检测跟踪ID切换造成静止时长累计错误", "护理操作暂停画面上传，患者实际已完成翻身"],
            "escalate": ["患者静止超过阈值，但可见轻微呼吸动作", "画面显示无活动，床垫信号存在少量变化", "患者侧卧时间较长，呼叫响应信息尚未返回", "摄像头部分遮挡，仅能观察到下肢静止", "夜间患者保持同一姿势，无法区分熟睡与意识异常"],
        },
        "advice": {
            "confirm": ["尽快床旁呼唤并评估意识及生命体征，必要时通知医生", "立即确认患者状态，协助调整体位并记录静止时长", "到场检查意识和呼吸，异常时按院内急症流程处理"],
            "reject": ["确认患者正常休息并核对设备状态，继续定时巡视", "检查视频流和跟踪状态，修正异常后恢复监测", "确认患者有正常活动迹象，记录误报并继续观察"],
            "escalate": ["尽快床旁确认呼叫反应，补查生命体征后再研判", "持续观察微动并人工复核，无法唤醒时立即升级", "结合床垫和连续画面复核，必要时增加巡视频率"],
        },
    },
    "night_wandering": {
        "cn": "夜间徘徊", "priority": "P2", "hours": [0, 1, 2, 3, 4, 5, 23],
        "rules": ["night_window=true", "bed_leave_duration>threshold", "repeated_walking=true"],
        "scenes": {
            "confirm": ["患者夜间离床后在床旁往返走动多次", "患者凌晨在病房与门区之间反复移动", "患者离床超过五分钟并持续无目的走动", "夜间患者多次靠近门口又返回床旁", "患者凌晨独自在走廊方向来回行走"],
            "reject": ["患者在护理人员陪同下按计划夜间步行训练", "患者起床如厕后直接返回床位，未出现反复走动", "值班人员经过床旁被错误关联到患者跟踪ID", "白天活动记录因时区错误被标记为夜间", "摄像头重复播放缓存片段形成虚假往返轨迹"],
            "escalate": ["患者夜间离床并有两次方向变化，轨迹随后丢失", "门区检测到重复人体，但身份关联不稳定", "患者在床旁停留较久，无法确认是否需要帮助", "低照度下出现间断行走轨迹，床垫确认离床", "患者凌晨多次坐起和下床，是否有人陪同未知"],
        },
        "advice": {
            "confirm": ["尽快陪同患者返回安全区域，了解需求并加强夜间巡视", "到场确认患者定向状态和步态安全，防止跌倒或走失", "引导患者安全回床，检查照明和呼叫器并记录情况"],
            "reject": ["确认活动有计划或已安全结束，修正时间和跟踪信息", "核对陪同记录与轨迹，确认无风险后关闭告警", "检查缓存与身份关联，排除误报后继续夜间监测"],
            "escalate": ["尽快确认患者位置和陪同情况，持续徘徊则立即干预", "补查门区轨迹并联系床旁，无法确认时升级处置", "加强观察并人工复核，优先防范跌倒和走失风险"],
        },
    },
    "door_departure": {
        "cn": "门区异常离开", "priority": "P2", "hours": [6, 9, 12, 16, 20, 23],
        "rules": ["door_sensor=open", "camera.person_departing=true"],
        "scenes": {
            "confirm": ["病房门打开后患者独自向门外移动", "患者离开床区并穿过门口，未检测到护理人员陪同", "门磁触发同时患者跟踪轨迹指向走廊", "患者推开病房门后持续离开监控区域", "夜间患者从门区离开，床垫同步报告离床"],
            "reject": ["护理人员开门进入病房，患者仍在床上", "送餐车经过门口触发人体误检，床垫显示患者在床", "门磁抖动产生打开信号，画面中无人通过", "家属获准陪同患者外出，护理记录已登记", "相邻病房人员经过被错误分配到当前节点"],
            "escalate": ["门磁报告打开且有人影经过，但身份无法确认", "患者轨迹靠近门口后被遮挡，是否离开未知", "床垫显示离床，门区画面仅捕获部分人体", "多人同时经过门口导致跟踪ID混淆", "夜间门被打开，低照度画面无法确认通过者"],
        },
        "advice": {
            "confirm": ["尽快确认患者去向并保障步行安全，必要时通知值班人员", "联系患者并检查走廊区域，确认是否需要陪同", "立即核实离开原因，防止走失并记录处置经过"],
            "reject": ["核对门磁、床垫和人员身份，确认误报后关闭事件", "检查门区跟踪与传感器状态，患者安全则继续观察", "确认外出已有陪同和记录，更新事件处置状态"],
            "escalate": ["尽快复核门区录像并确认患者是否仍在病房", "联系床旁和走廊值班人员，身份不明时升级查找", "结合床垫与门区轨迹核实，无法确认则按离开处理"],
        },
    },
    "bedsore_risk": {
        "cn": "压疮风险", "priority": "P3", "hours": [1, 5, 8, 13, 17, 22],
        "rules": ["position_duration>=7200", "bed_sensor.on_bed=true"],
        "scenes": {
            "confirm": ["患者同一卧位持续超过两小时，压力分布集中且未翻身", "患者长时间仰卧，床垫未记录有效体位变化", "高风险患者连续卧床超过护理计划规定时长", "患者骶尾区域压力持续偏高，翻身提醒已超时", "夜间患者保持单侧卧位超过两小时且无护理记录"],
            "reject": ["护理人员已完成翻身，但床垫数据尚未刷新", "患者自主小幅调整体位，跟踪系统未正确累计", "空床测试产生长时压力信号，床位当前无人", "传感器漂移导致压力分布保持不变，画面显示已翻身", "护理计划允许当前体位且皮肤评估记录正常"],
            "escalate": ["床垫提示同一体位超时，但画面显示轻微侧移", "翻身记录缺失，无法确认是否已由护理人员处理", "压力集中时间接近阈值，患者活动能力信息未知", "床垫部分分区失效，体位变化判断不完整", "夜间长期卧位告警出现，但最近护理记录尚未同步"],
        },
        "advice": {
            "confirm": ["按护理计划协助翻身并检查受压皮肤，记录处理时间", "尽快调整体位和减压，评估皮肤并更新翻身记录", "协助患者减压，检查高风险部位并加强定时巡视"],
            "reject": ["核对翻身记录和床垫数据，确认已处理后校准监测", "确认患者体位已变化，修正累计时长并继续观察", "检查传感器分区和护理记录，排除误报后关闭提醒"],
            "escalate": ["人工核对最近翻身与皮肤状态，必要时提前减压", "补查护理记录和床垫分区，无法确认时先协助翻身", "尽快评估受压风险并完善记录，持续超时则安排处理"],
        },
    },
    "environment_anomaly": {
        "cn": "环境异常", "priority": "P3", "hours": [0, 6, 10, 14, 18, 22],
        "rules": ["environment.value_out_of_range=true"],
        "scenes": {
            "confirm": ["病房二氧化碳浓度持续超过设定阈值", "室温连续升高并超过护理环境上限", "病房照度在非睡眠时段持续过低", "湿度长时间超出设备安全范围", "夜间病房温度异常下降且连续多次测量一致"],
            "reject": ["传感器维护时使用测试气体造成短时高读数", "环境探头靠近出风口导致局部温度异常", "设备刚启动尚未完成校准，其他探头读数正常", "清洁遮挡光照传感器造成短时低照度", "历史缓存数据被重复上报，当前环境数值正常"],
            "escalate": ["单个探头报告二氧化碳偏高，其他区域数据缺失", "温度在阈值附近反复波动，持续时间尚不足", "照度异常与夜间模式配置存在冲突", "湿度读数快速变化，但传感器质量标记一般", "病房环境告警持续出现，备用探头暂未上线"],
        },
        "advice": {
            "confirm": ["检查通风和温控设备，必要时调整环境并持续复测", "核实病房环境并采取通风或温控措施，记录处理结果", "尽快排查环境设备，确保患者舒适安全后继续监测"],
            "reject": ["确认当前环境正常，校准探头并标记维护产生的误报", "检查传感器位置与缓存时间，修正后继续采集", "核对备用测量结果，无异常后关闭提醒并记录原因"],
            "escalate": ["使用备用设备复测并观察趋势，持续异常时安排处理", "核对多点环境数据和设备状态，无法排除时通知维护", "先人工测量确认，接近阈值时加强监测并准备调整"],
        },
    },
    "device_fault": {
        "cn": "设备故障", "priority": "P3", "hours": [2, 7, 11, 15, 19, 23],
        "rules": ["sensor.quality.degraded=true", "degraded_seconds>=60"],
        "scenes": {
            "confirm": ["床垫传感器连续失联超过一分钟且无新数据", "摄像头视频流反复中断并报告设备错误", "环境探头持续输出非法数值，质量标记为降级", "呼叫器自检失败且无法上报按键状态", "边缘节点检测到摄像头供电异常并停止采集"],
            "reject": ["设备按计划重启并已恢复，短时离线属于维护窗口", "网络切换造成一次丢包，传感器随后连续上报正常", "模拟演练主动设置了质量降级标记", "采样间隔调整被误认为数据中断", "旧版健康消息延迟到达，设备当前自检正常"],
            "escalate": ["床垫数据间歇丢失，连接状态在正常与降级间切换", "摄像头画面可见但关键点连续为空", "环境探头出现少量异常值，是否硬件故障未知", "呼叫器偶发无响应，重试后暂时恢复", "节点报告设备错误但备用传感器仍在工作"],
        },
        "advice": {
            "confirm": ["启用备用监测并通知设备维护，故障排除前加强人工巡视", "检查供电和连接，切换备用设备并登记故障工单", "立即确认监测盲区，安排维修并采取人工替代措施"],
            "reject": ["核对维护窗口和当前自检结果，恢复后关闭误报", "确认设备连续上报正常，记录瞬时通信异常", "检查消息时间戳和采样配置，无故障后继续监测"],
            "escalate": ["复测设备连接和数据质量，异常持续时切换备用监测", "检查供电、网络及探头状态，无法定位时通知维护", "暂时加强人工巡视并观察设备，重复失败则升级处理"],
        },
    },
    "node_offline": {
        "cn": "节点失联", "priority": "P3", "hours": [0, 5, 9, 14, 18, 23],
        "rules": ["health_heartbeat_timeout=true"],
        "scenes": {
            "confirm": ["边缘节点连续多个心跳周期未响应，MQTT连接已断开", "节点健康状态转为离线且最后上报时间超过阈值", "边缘代理进程停止，摄像头和床垫数据同时中断", "节点网络不可达并停止补传本地事件", "夜间边缘节点失联超过五分钟，所有观测均停止"],
            "reject": ["云端短时网络抖动导致心跳延迟，节点本地运行正常", "节点滚动重启已完成并恢复连续心跳", "监控服务时间不同步造成超时误判", "MQTT代理切换期间丢失一次心跳，观测数据未中断", "演练脚本主动模拟离线，节点实际连接正常"],
            "escalate": ["节点心跳间歇超时，但仍能收到少量缓存事件", "健康主题停止更新，观测主题仍有延迟数据", "节点状态在离线与降级之间反复切换", "MQTT连接存在但最近心跳内容不完整", "夜间节点超过两个周期无心跳，网络探测结果不稳定"],
        },
        "advice": {
            "confirm": ["通知维护人员检查节点和网络，恢复前加强对应床位巡视", "确认边缘供电与进程状态，启用人工监测并尽快恢复", "检查节点连接和本地缓存，修复后确认事件补传完整"],
            "reject": ["核对节点实时心跳和观测流，确认恢复后关闭告警", "检查时钟和代理切换记录，无中断后标记误报", "确认节点本地功能正常，修正健康超时配置"],
            "escalate": ["继续探测节点并人工确认床位安全，持续失联则通知维护", "核对各主题最后时间，无法确认时先启用人工巡视", "检查网络与进程状态，重复超时则按节点离线处理"],
        },
    },
}


EVIDENCE = {
    "confirm": ["摄像头与床垫结果一致", "连续观测支持本次告警", "两种传感器均给出异常信号", "异常状态持续超过规则阈值", "前后帧与传感器时间戳一致"],
    "reject": ["连续复核发现告警与现场状态不一致", "第二观测源未支持本次告警", "异常只出现于单帧或单一设备", "现场状态可由正常护理活动解释", "复核记录显示信号来源并非患者异常"],
    "escalate": ["不同观测源之间存在冲突", "关键区域存在遮挡且证据不完整", "异常持续时间处于阈值附近", "设备质量一般且需人工补充信息", "后续观测暂时缺失，不能排除风险"],
}

CONFIDENCE_RANGES = {
    "confirm": (0.64, 0.98),
    "reject": (0.12, 0.68),
    "escalate": (0.38, 0.82),
}


def judgment_sequence(n: int, rng: random.Random) -> List[str]:
    counts = {"confirm": n // 2, "reject": n // 4}
    counts["escalate"] = n - counts["confirm"] - counts["reject"]
    values = [name for name, count in counts.items() for _ in range(count)]
    rng.shuffle(values)
    return values


def event_details(event_type: str, judgment: str, index: int, scene: str) -> Dict[str, Any]:
    good = judgment == "confirm"
    ambiguous = judgment == "escalate"
    details: Dict[str, Any] = {
        "evidence_note": scene,
        "observation_quality": "degraded" if ambiguous else "good",
    }
    if event_type == "fall_suspected":
        details.update(posture="falling" if good else "unknown", fall_score=round(0.78 if good else 0.46 + (index % 3) * 0.08, 2), on_bed=not good)
    elif event_type == "seizure":
        details.update(tremor_score=round(0.86 if good else 0.42 + (index % 4) * 0.09, 2), repetitive_motion=good or ambiguous)
    elif event_type == "fall_prediction":
        details.update(posture="lying_edge" if good or ambiguous else "lying_center", fall_score=round(0.81 if good else 0.48 + (index % 3) * 0.09, 2), on_bed=True)
    elif event_type == "nurse_call":
        details.update(call_requested=good, signal_count=1 + index % 3)
    elif event_type in ("bed_leave", "night_wandering", "door_departure"):
        details.update(absence_seconds=35 + index % 150, on_bed=not good, door_open=event_type != "bed_leave")
    elif event_type in ("long_still", "bedsore_risk"):
        base = 7600 if event_type == "bedsore_risk" else 360
        details.update(position_duration=base + index * 7, on_bed=True)
    elif event_type == "abnormal_posture":
        details.update(posture="grabbing_chest" if good else "leaning", position_duration=8 + index % 40)
    elif event_type == "environment_anomaly":
        details.update(temperature=round(29.0 + index % 4, 1), co2=900 + (index % 7) * 160)
    elif event_type == "device_fault":
        details.update(sensor="camera" if index % 2 else "bed_sensor", degraded_seconds=55 + index % 90)
    elif event_type == "node_offline":
        details.update(heartbeat_age_seconds=90 + index % 360, buffered_events=index % 8)
    return details


def make_description(ward: str, bed: str, time_text: str, scene: str, evidence: str, confidence: float, variant: int) -> str:
    patterns = [
        f"{ward}病区{bed}于{time_text}{scene}；{evidence}，系统置信度{confidence:.0%}",
        f"{time_text}，{ward}病区{bed}{scene}。{evidence}，本次检测置信度{confidence:.0%}",
        f"系统在{time_text}发现{bed}{scene}；{evidence}，融合置信度为{confidence:.0%}",
        f"{bed}在{time_text}{scene}。监测摘要显示{evidence}，置信度{confidence:.0%}",
    ]
    return patterns[variant % len(patterns)]


def build_prompt(record: Dict[str, Any]) -> str:
    spec = EVENTS[record["event_type"]]
    ctx = record["context"]
    detail_text = "; ".join(f"{k}={v}" for k, v in ctx["details"].items())
    return (
        f"床位{ctx['bed']}检测到【{spec['cn']}】事件。\n"
        f"置信度: {ctx['confidence']:.0%}，优先级: {record['labels']['priority']}\n"
        f"触发规则: {', '.join(ctx['rule_hits'])}\n"
        f"详细数据: {detail_text}\n"
        "请给出一句话状况描述和处置建议。"
    )


def cloud_prompt(record: Dict[str, Any]) -> str:
    ctx = record["context"]
    return (
        "你是智慧病房的AI护理助手。边缘端报告了一起安全事件，请研判其真实性。\n\n"
        f"事件类型: {record['event_type']}\n优先级: {record['labels']['priority']}\n"
        f"置信度: {ctx['confidence']}\n床位: {ctx['bed']}\n"
        f"详情: {json.dumps(ctx['details'], ensure_ascii=False, sort_keys=True)}\n"
        f"自然语言摘要: {record['description']}\n\n"
        "请给出研判结果，格式为: judgment|confidence|advice\n"
        "judgment 取值为 confirm(确认)/reject(误报)/escalate(升级)。"
    )


def generate_records() -> List[Dict[str, Any]]:
    rng = random.Random(SEED)
    records: List[Dict[str, Any]] = []
    serial = 1
    for event_type, total in COUNTS.items():
        spec = EVENTS[event_type]
        test_n = total // 5
        for split, n, scene_indexes, start_date in (
            ("train", total - test_n, [0, 1, 2, 3], datetime(2026, 8, 1, tzinfo=timezone.utc)),
            ("test", test_n, [4], datetime(2026, 8, 20, tzinfo=timezone.utc)),
        ):
            judgments = judgment_sequence(n, rng)
            for local_i, judgment in enumerate(judgments):
                scene_index = scene_indexes[local_i % len(scene_indexes)]
                scene = spec["scenes"][judgment][scene_index]
                evidence = EVIDENCE[judgment][(local_i + scene_index) % len(EVIDENCE[judgment])]
                low, high = CONFIDENCE_RANGES[judgment]
                confidence = round(rng.uniform(low, high), 2)
                bed = f"B{1 + (serial * 7 + local_i) % 3:02d}"
                hour = spec["hours"][(local_i + scene_index) % len(spec["hours"])]
                minute = (serial * 13 + local_i * 7) % 60
                occurred = start_date + timedelta(days=(local_i * 3 + serial) % 16, hours=hour, minutes=minute)
                time_text = f"{hour:02d}:{minute:02d}"
                advice_options = spec["advice"][judgment]
                advice = advice_options[(local_i + scene_index) % len(advice_options)]
                details = event_details(event_type, judgment, local_i, scene)
                record = {
                    "id": f"nlu-{serial:04d}",
                    "event_type": event_type,
                    "description": make_description("W-01", bed, time_text, scene, evidence, confidence, local_i),
                    "context": {
                        "ward": "W-01",
                        "bed": bed,
                        "node_id": f"EDGE-W01-{bed}",
                        "confidence": confidence,
                        "occurred_at": occurred.isoformat().replace("+00:00", "Z"),
                        "network_state": ["connected", "degraded", "disconnected"][(serial + local_i) % 3],
                        "rule_hits": list(spec["rules"]),
                        "details": details,
                        "scenario_family": f"{event_type}-{judgment}-{'test' if split == 'test' else f'train-{scene_index + 1}'}",
                    },
                    "labels": {
                        "priority": spec["priority"],
                        "urgency": URGENCY[spec["priority"]],
                        "advice": advice,
                        "judgment": judgment,
                    },
                    "source": "manual",
                    "split": split,
                }
                records.append(record)
                serial += 1
    return records


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def to_sft(record: Dict[str, Any], task: str) -> Dict[str, Any]:
    labels = record["labels"]
    ctx = record["context"]
    if task == "event_enhancement":
        user = build_prompt(record)
        status = {"confirm": "信号明确", "reject": "疑似误报", "escalate": "证据冲突"}[labels["judgment"]]
        event_cn = EVENTS[record["event_type"]]["cn"]
        assistant = f"【{labels['urgency']}】{ctx['bed']}{event_cn}{status}。{labels['advice']}"
        if len(assistant) > 60:
            raise ValueError(f"event enhancement target exceeds 60 characters: {record['id']}")
    else:
        user = cloud_prompt(record)
        assistant = f"{labels['judgment']}|{ctx['confidence']:.2f}|{labels['advice']}"
    return {
        "id": f"{record['id']}-{task}",
        "task": task,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def to_inference_request(record: Dict[str, Any]) -> Dict[str, Any]:
    ctx = record["context"]
    event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "smart-ward/" + record["id"]))
    trace_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "smart-ward/trace/" + record["id"]))
    return {
        "dataset_id": record["id"],
        "expected": dict(record["labels"]),
        "request": {
            "event_id": event_id,
            "trace_id": trace_id,
            "request_mode": "hybrid",
            "timeout_ms": 30000,
            "requested_at": ctx["occurred_at"],
            "event_type": record["event_type"],
            "priority": record["labels"]["priority"],
            "confidence": ctx["confidence"],
            "ward_id": ctx["ward"],
            "node_id": ctx["node_id"],
            "bed_id": ctx["bed"],
            "model_name": "qwen2.5-1.5b-instruct",
            "model_version": "dataset-eval",
            "details": ctx["details"],
            "llm_prompt": cloud_prompt(record),
        },
    }


def write_schema(path: Path) -> None:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SmartWardNLUSample",
        "type": "object",
        "required": ["id", "event_type", "description", "context", "labels", "source", "split"],
        "properties": {
            "id": {"type": "string", "pattern": "^nlu-[0-9]{4}$"},
            "event_type": {"type": "string", "enum": list(COUNTS)},
            "description": {"type": "string", "minLength": 10},
            "context": {
                "type": "object",
                "required": ["ward", "bed", "confidence", "occurred_at", "network_state", "rule_hits", "details", "scenario_family"],
                "properties": {
                    "ward": {"type": "string"}, "bed": {"type": "string"}, "node_id": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "occurred_at": {"type": "string", "format": "date-time"},
                    "network_state": {"enum": ["connected", "degraded", "disconnected"]},
                    "rule_hits": {"type": "array", "items": {"type": "string"}},
                    "details": {"type": "object"}, "scenario_family": {"type": "string"},
                },
            },
            "labels": {
                "type": "object", "required": ["priority", "urgency", "advice", "judgment"],
                "properties": {
                    "priority": {"enum": ["P1", "P2", "P3"]},
                    "urgency": {"enum": ["紧急", "警告", "提醒"]},
                    "advice": {"type": "string", "minLength": 8},
                    "judgment": {"enum": ["confirm", "reject", "escalate"]},
                },
            },
            "source": {"enum": ["manual", "real_event", "diffusion"]},
            "split": {"enum": ["train", "test"]},
        },
        "additionalProperties": False,
    }
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/ward-nlu-500-v1"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(exist_ok=True)

    records = generate_records()
    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    assert len(records) == 500 and len(train) == 400 and len(test) == 100

    write_jsonl(out / f"nlu-dataset-v{VERSION}.jsonl", records)
    (out / f"nlu-dataset-v{VERSION}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(out / f"train-v{VERSION}.jsonl", train)
    write_jsonl(out / f"test-v{VERSION}.jsonl", test)
    write_jsonl(out / f"qwen-sft-train-v{VERSION}.jsonl", (to_sft(r, task) for r in train for task in ("event_enhancement", "cloud_judgment")))
    write_jsonl(out / f"qwen-sft-test-v{VERSION}.jsonl", (to_sft(r, task) for r in test for task in ("event_enhancement", "cloud_judgment")))
    write_jsonl(out / f"project-inference-test-v{VERSION}.jsonl", (to_inference_request(r) for r in test))
    write_schema(out / "schema-v1.0.0.json")

    manifest = {
        "name": "ward-nlu-500",
        "version": VERSION,
        "seed": SEED,
        "total": len(records),
        "train": len(train),
        "test": len(test),
        "event_distribution": dict(Counter(r["event_type"] for r in records)),
        "judgment_distribution": dict(Counter(r["labels"]["judgment"] for r in records)),
        "source_distribution": dict(Counter(r["source"] for r in records)),
        "derived_sft_train": len(train) * 2,
        "derived_sft_test": len(test) * 2,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
