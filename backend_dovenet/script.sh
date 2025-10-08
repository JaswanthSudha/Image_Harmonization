python test.py --dataset_root "C:\Users\JaswanthSudha\Desktop\Image-Harmonization-Dataset-iHarmony4\custom_data" --name experiment_name_pretrain --model dovenet --dataset_mode iharmony4 --netG s2ad --is_train 0 --norm batch --no_flip --preprocess none --num_test 1 --gpu_ids -1
#inference 
python test_inference.py --dataset_root "path/to/custom_data" --name experiment_name_pretrain --model dovenet --dataset_mode inference --netG s2ad --is_train 0 --norm batch --no_flip --preprocess none --num_test 1 --gpu_ids -1

#inference 4k 
python patch_based_4k.py --dataset_root "C:\Users\JaswanthSudha\Desktop\Image-Harmonization-Dataset-iHarmony4\custom_data" --name experiment_name_pretrain --model dovenet --dataset_mode inference --netG s2ad --is_train 0 --norm batch --gpu_ids -1

python test_inference.py --dataset_root "C:\Users\JaswanthSudha\Desktop\Image-Harmonization-Dataset-iHarmony4\custom_data" --name experiment_name_pretrain --model dovenet --dataset_mode inference --netG s2ad --is_train 0 --norm batch --no_flip --preprocess none  --gpu_ids -1