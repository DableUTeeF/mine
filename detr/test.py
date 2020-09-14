from datasets.coco import CocoCustom

a = CocoCustom('/media/palm/BiggerData/mine/new/i',
               '/home/palm/PycharmProjects/mine/anns/train.json',
               None,
               False)
x = a[0]
print(x)
