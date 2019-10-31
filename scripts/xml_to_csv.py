import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# def main():
    # image_path = os.path.join(os.getcwd(), 'annotations')
    # xml_df = xml_to_csv(image_path)
    # xml_df.to_csv('corn_labels.csv', index=None)
    # print('Successfully converted xml to csv.')


# New:
def main():
    if len(sys.argv) != 2:
        print("format: python xml_to_csv.py xml_path")
        return -1
    else:
        annotation_path=sys.argv[1]
        print("Using xml path:", annotation_path)

    for folder in ['train', 'test']:
        image_path = os.path.join(annotation_path, folder)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv((annotation_path+folder+'_labels.csv'), index=None)
        print('Successfully converted xml to csv.')

main()
