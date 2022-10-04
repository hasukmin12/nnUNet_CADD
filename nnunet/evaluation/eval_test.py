
# https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/common_questions.md


from evaluator import evaluate_folder


folder_with_gt = '/vast/AI_team/sukmin/nnUNet_raw_data_base/nnUNet_raw_data/Task310_Urinary/labelsTs'
# folder_with_gt = '/data5/sukmin/nnUNet_raw_data_base/nnUNet_raw_data/Task301_KiPA/labelsTs'
# folder_with_gt = '/data5/sukmin/nnUNet_raw_data_base/nnUNet_raw_data/Task211_Ureter/labelsTs'
# folder_with_gt = '/data/sukmin/nnUNet_raw_data_base/nnUNet_raw_data/Task150_Bladder/labelsTs'
# folder_with_gt = '/data/sukmin/_has_Bladder_red'

folder_with_pred = '/home/sukmin/datasets/inf_2_GT_310_UNet_nnUNetTrainer'

labels = (0,1) # test 하고 싶은 라벨 입

evaluate_folder(folder_with_gt, folder_with_pred, labels)



# #
#
# folder_with_gt = '/data5/sukmin/nnUNet_raw_data_base/nnUNet_raw_data/Task240_Ureter/labelsTs'
# folder_with_pred = '/data5/sukmin/inf_2_GT_261_color_LR_half'
#
# labels = (0,1,2,3,4) # test 하고 싶은 라벨 입
#
# evaluate_folder(folder_with_gt, folder_with_pred, labels)