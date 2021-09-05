#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2021/6/19 15:58
# @Author  :Jiawei Lian
# @FileName: defect_detector
# @Software: PyCharm

result = {
    "result": [
        {
            "boxes": [
                [
                    1037.7069091796875,
                    338.4924621582031,
                    1104.1451416015625,
                    350.7931213378906
                ],
                [
                    1239.7542724609375,
                    345.3689880371094,
                    1254.3543701171875,
                    436.52642822265625
                ],
                [
                    1041.7928466796875,
                    343.25054931640625,
                    1099.8563232421875,
                    352.453125
                ],
                [
                    1073.7562255859375,
                    790.2770385742188,
                    1085.835693359375,
                    797.633544921875
                ],
                [
                    1209.29248046875,
                    754.3037109375,
                    1227.1668701171875,
                    792.6311645507812
                ],
                [
                    1075.0316162109375,
                    791.220947265625,
                    1082.1363525390625,
                    796.9646606445312
                ],
            ],
            "labels": [
                1,
                2,
                1,
                4,
                2,
                4,
            ],
            "scores": [
                0.9731950163841248,
                0.3323192000389099,
                0.2229144424200058,
                0.2062360942363739,
                0.1661888062953949,
                0.10665187239646912,
            ]
        }
    ]
}
print(result)
idxes = []
for idx in range(len(result['result'][0]['scores'])):
    if result['result'][0]['scores'][idx] < 0.2 and result['result'][0]['labels'][idx] == 2:
        idxes.append(idx)
print(idxes)
for i in idxes:
    # result['result'][0]['scores'].pop()
    # result['result'][0]['labels'].pop()
    # result['result'][0]['boxes'].pop()
    result['result'][0]['scores'].pop()
    result['result'][0]['labels'].pop()
    result['result'][0]['boxes'].pop()
print(result)
# print(idx)
#     result[idx]['boxes'] = result[idx][
#         'boxes'].cpu().detach().numpy().tolist()
#     result[idx]['labels'] = result[idx][
#         'labels'].cpu().detach().numpy().tolist()
#     result[idx]['scores'] = result[idx][
#         'scores'].cpu().detach().numpy().tolist()
#
# result = {"result": result}
#
# print(result)
