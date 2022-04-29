#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data


if __name__ == "__main__":
    """
    This is the KiPA from sukmin Ha
    """

    base = "/data5/sukmin/KiPA/train"

    task_id = 300
    task_name = "KiPA"
    foldername = "Task%03.0d_%s" % (task_id, task_name)

    nnUNet_raw_data = '/data5/sukmin/nnUNet_raw_data_base/nnUNet_raw_data'

    out_base = join(nnUNet_raw_data, foldername)
    # out_base = join(base, foldername)


    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_patient_names = []
    test_patient_names = []
    img_base = join(base, 'image')
    seg_base = join(base, 'label')

    img_list = next(os.walk(img_base))[2]
    seg_list = next(os.walk(seg_base))[2]
    img_list.sort()
    seg_list.sort()

    train_list = img_list[:50]
    test_list = img_list[50:]


    for p in train_list:
        label_file = join(seg_base, p)
        image_file = join(img_base, p)
        name = "case_{0:05d}".format(int(img_list.index(p)))
        shutil.copy(image_file, join(imagestr, name + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, name + ".nii.gz"))
        train_patient_names.append(name)

    # 나중에 test inference를 위해 폴더는 만들어놓
    for p in test_list:
        label_file = join(seg_base, p)
        image_file = join(img_base, p)
        name = "case_{0:05d}".format(int(img_list.index(p)))
        shutil.copy(image_file, join(imagests, name + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelsts, name + ".nii.gz"))
        test_patient_names.append(name)





    json_dict = {}
    json_dict['name'] = "KiPA"
    json_dict['description'] = "KiPA challenge"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "KiPA dataset in nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Renal vein",
        "2": "Kidney",
        "3": "Renal artery",
        "4": "Tumor",
    }

    json_dict['numTraining'] = len(train_patient_names)
    # json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTs/%s.nii.gz" % i.split("/")[-1]} for i in
    #                    test_patient_names]
    save_json(json_dict, os.path.join(out_base, "dataset.json"))