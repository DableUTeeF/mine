import os
from xml.etree import cElementTree as ET


def created(full_path):
    root = ET.Element('annotation')
    ET.SubElement(root, 'path').text = full_path
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = '1920'
    ET.SubElement(size, 'height').text = '1080'
    o = ET.SubElement(root, 'object')
    ET.SubElement(o, 'name').text = 'car180'
    bbox = ET.SubElement(o, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = '1142'
    ET.SubElement(bbox, 'ymin').text = '514'
    ET.SubElement(bbox, 'xmax').text = '1486'
    ET.SubElement(bbox, 'ymax').text = '988'
    tree = ET.ElementTree(root)
    return tree


if __name__ == '__main__':
    path = '/media/palm/BiggerData/mine/PU_22788397_00_20200430_031829_BKQ02/annotations'
    for file in range(2580, 6060, 60):
        file = str(file) + '.xml'
        if os.path.exists(os.path.join(path, file)):
            tree = ET.parse(os.path.join(path, file))
            xmlRoot = tree.getroot()
            o = ET.SubElement(xmlRoot, 'object')
            ET.SubElement(o, 'name').text = 'car180'
            bbox = ET.SubElement(o, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = '1142'
            ET.SubElement(bbox, 'ymin').text = '514'
            ET.SubElement(bbox, 'xmax').text = '1486'
            ET.SubElement(bbox, 'ymax').text = '988'
            tree = ET.ElementTree(xmlRoot)
            tree.write(os.path.join('anns/new_xml', file))
        else:
            dummy_tree = created(f'/media/palm/BiggerData/mine/PU_22788397_00_20200430_031829_BKQ02/images/{file}.jpg')
            dummy_tree.write(os.path.join('anns/new_xml', file))

