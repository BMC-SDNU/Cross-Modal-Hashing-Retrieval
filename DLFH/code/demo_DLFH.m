
function [] = demo_DLFH(bits, dataname)
    addpath('../../Data');
    addpath('utils');
	bits = str2num(bits);
    if dataname == 'flickr'

        load('mir_cnn.mat');
    elseif dataname == 'nuswide'
        load('nus_cnn.mat');
    elseif dataname == 'coco'
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end

    param.bits = bits;
    param.num_samples = param.bits;
    param.maxIter = 30;
    param.gamma = 1e-2;
    param.lambda = 8;
    % param.dataset_name = dataset_name;

    dataset.XDatabase = I_tr;
    dataset.YDatabase = T_tr;
    dataset.XTest = I_te;
    dataset.YTest = T_te;
    dataset.learn_dataX = I_db;
    dataset.learn_dataT = T_db;
    dataset.databaseL = L_tr;
    dataset.L_te = L_te;
    dataset.L_db = L_db;

    [MAP_I2T, MAP_T2I] = DLFH_algo(dataset, param);

	result_I2T = sprintf('%3d-%s, I2T MAP = %.4f\n', bits, dataname, MAP_I2T);
	result_T2I = sprintf('%3d-%s, T2I MAP = %.4f\n', bits, dataname, MAP_T2I);

	fprintf(result_I2T);
	fprintf(result_T2I);
	
	name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, result_I2T);
    fprintf(fid, result_T2I);
	fclose(fid);

end

