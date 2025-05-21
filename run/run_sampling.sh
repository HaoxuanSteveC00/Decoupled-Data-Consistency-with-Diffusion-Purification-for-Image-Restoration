#FFHQ
python dcdp.py --task_config=./task_configurations/gaussian_deblur_config.yaml --purification_config=./purification_configurations/purification_config_gaussian_deblur.yaml \
                 --model_config=./model_configurations/model_config_ffhq.yaml

python dcdp.py --task_config=./task_configurations/motion_deblur_config.yaml --purification_config=./purification_configurations/purification_config_motion_deblur.yaml \
                 --model_config=./model_configurations/model_config_ffhq.yaml

python dcdp.py --task_config=./task_configurations/super_resolution_config.yaml --purification_config=./purification_configurations/purification_config_super_resolution.yaml \
                 --model_config=./model_configurations/model_config_ffhq.yaml

python dcdp.py --task_config=./task_configurations/inpainting_config.yaml --purification_config=./purification_configurations/purification_config_inpainting.yaml \
                 --model_config=./model_configurations/model_config_ffhq.yaml 

#ImageNet
python dcdp.py --task_config=./task_configurations/gaussian_deblur_config.yaml --purification_config=./purification_configurations/purification_config_gaussian_deblur_imagenet.yaml \
                 --model_config=./model_configurations/model_config_ImageNet.yaml

python dcdp.py --task_config=./task_configurations/motion_deblur_config.yaml --purification_config=./purification_configurations/purification_config_motion_deblur_imagenet.yaml \
                 --model_config=./model_configurations/model_config_ImageNet.yaml

python dcdp.py --task_config=./task_configurations/super_resolution_config.yaml --purification_config=./purification_configurations/purification_config_super_resolution_imagenet.yaml \
                 --model_config=./model_configurations/model_config_ImageNet.yaml

python dcdp.py --task_config=./task_configurations/inpainting_config.yaml --purification_config=./purification_configurations/purification_config_inpainting_imagenet.yaml \
                 --model_config=./model_configurations/model_config_ImageNet.yaml