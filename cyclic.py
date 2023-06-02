import datetime
import os,shutil
import datetime
import tqdm
import fix
for repeat in range(5):
    now = datetime.datetime.now()
    LRPTS_dirpath = "logs/LRPTS{:%Y%m%dT%H%M}/".format(now)
    os.mkdir(LRPTS_dirpath)
    for loop in range(5):
        print('_____________________________________________LOOP{}________________________________________________'.format(loop))
        GT2 = r'../../../datasets/MoNuSACGTHV'.format(loop)
        try:
            shutil.rmtree(GT2)
        except:
            pass
        shutil.copytree(r'../../../datasets/MoNuSACGT', GT2)
        for fname in os.listdir(GT2 + '/stage1_train/'):
            stage = GT2 + '/stage1_train/'
            imsdirpath = stage + '/' + fname + '/masks/'
            imsdir = os.listdir(imsdirpath)
            length = len(imsdir)
            count = length
            used_nums = 160+160*loop
            for im in imsdir:
                if count > used_nums:
                    os.remove(imsdirpath + im)
                    count -= 1
                # k=random.randint(0,100)
                # if k>30+loop*10 :
                #     os.remove(imsdirpath+im)
        # shutil.copytree(GT2, LRPTS_dirpath + 'LOOP{}_Input_Dataset'.format(loop))
        fix.fix(LRPTS_dirpath+'Loop{}_Input_250seedcheck'.format(loop))
        shutil.copytree('../../../datasets/MoNuSAC/stage1_train',LRPTS_dirpath + 'LOOP{}_Input_Dataset/'.format(loop))
        python='/home/iftwo/anaconda3/envs/hovernet/bin/python'
        os.system('nohup {} extract_patches_MONU.py '.format(python))
        shutil.copytree("/data1/wyj/M/datasets/MoNuSAC/HV0/masks-new",LRPTS_dirpath + 'LOOP{}_Input_PREPROCESSED_DATASET'.format(loop))
        os.system('{} run_train.py --gpu=0'.format(python))
        shutil.copytree('logs/00', LRPTS_dirpath + 'LOOP{}'.format(loop))
        for sid in range(5):
            os.system('nohup {} run_infer.py --gpu=1 --nr_types=5 --batch_size=8 --model_mode=original '
                      '--model_path=logs/00/net_epoch={}0.tar --type_info_path=type_info.json tile --input_dir=/data1/wyj/M/d'
                      'atasets/MoNuSACCROP/images/ --output_dir=./outputtestourtemp/ --mem_usage=0.1 --draw_dot --save_qupath --save_raw_map'.format(python,sid+1))
            shutil.copytree('./outputtestourtemp', LRPTS_dirpath + 'LOOP{}_output_of_generation{}'.format(loop,sid))