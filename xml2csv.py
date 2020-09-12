from xml.etree import cElementTree as ET
import os


if __name__ == '__main__':
    open('anns/val_ann.csv', 'w')
    open('anns/c.csv', 'w')
    classes = []
    root = '/media/palm/BiggerData/mine/new/a/'
    paths = ['PU_23550891_00_20200905_203537_BKQ02-005.mkv',
             'PU_23550891_00_20200905_214516_BKQ02-003.mkv',
             'PU_23550891_00_20200905_230000_BKQ02.mkv',
             'PU_23550891_00_20200906_000001_BKQ02.mkv',
             ]

    with open('anns/ann.csv', 'w') as wr:
        for p in paths:
            path = os.path.join(root, p)
            for file in os.listdir(path):
                tree = ET.parse(os.path.join(path, file))
                if len(tree.findall('object')) == 0:
                    continue
                ln = ''
                cls = ''
                xmin = 0
                xmax = 0
                ymin = 0
                ymax = 0
                impath = ''
                for elem in tree.iter():
                    if 'path' in elem.tag:
                        impath = elem.text
                    if 'object' in elem.tag:
                        if cls != '' and (xmax+xmin+ymax+ymax) != 0 and impath != 0 and cls != 'ladder':
                            if cls not in classes:
                                with open('anns/c.csv', 'a') as cwr:
                                    cwr.write(f'{cls},{len(classes)}\n')
                                classes.append(cls)
                            ln = f'{impath},{xmin},{ymin},{xmax},{ymax},{cls}'
                            wr.write(ln)
                            wr.write('\n')
                    elif 'name' in elem.tag:
                        cls = elem.text
                        if 'car' in cls:
                            cls = 'car'
                    elif 'xmin' in elem.tag:
                        xmin = elem.text
                    elif 'ymin' in elem.tag:
                        ymin = elem.text
                    elif 'xmax' in elem.tag:
                        xmax = elem.text
                    elif 'ymax' in elem.tag:
                        ymax = elem.text
                if 1: # cls != 'obj':
                    if cls not in classes:
                        with open('anns/c.csv', 'a') as cwr:
                            cwr.write(f'{cls},{len(classes)}\n')
                        classes.append(cls)
                    ln = f'{impath},{xmin},{ymin},{xmax},{ymax},{cls}'
                    wr.write(ln)
                    wr.write('\n')
