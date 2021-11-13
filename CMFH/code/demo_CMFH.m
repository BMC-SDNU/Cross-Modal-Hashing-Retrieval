
function [] = demo_CMFH(bits, dataname)
    %addpath('../common');

    addpath('../../Data');
	bits = str2num(bits);

    CMFHparam.alphas = [0.5 0.5];
    CMFHparam.gamma = 0.01;
    CMFHparam.mu = 100;
    if dataname == 'flickr'
        load('mir_cnn.mat');
    elseif dataname == 'nuswide'
        load('nus_cnn.mat');
    elseif dataname == 'coco'
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end

    %% centralization
    %fprintf('centralizing data...\n');
    I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
    I_db = bsxfun(@minus, I_db, mean(I_tr, 1));
    I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));

    T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
    T_db = bsxfun(@minus, T_db, mean(T_tr, 1));
    T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

    I_temp = I_tr';
    T_temp = T_tr';
    [row, col]= size(I_temp);  
    [rowt, colt] = size(T_temp);

    I_temp = bsxfun(@minus,I_temp , mean(I_temp,2));
    T_temp = bsxfun(@minus,T_temp, mean(T_temp,2));
    Im_te = (bsxfun(@minus, I_te', mean(I_temp, 2)))';
    Te_te = (bsxfun(@minus, T_te', mean(T_temp, 2)))';

    [model, B]= CMFH_learn(I_temp', T_temp',CMFHparam, bits);
    %% calculate hash codes
     [codeDbX,codeDbY,test_hash_X, test_hash_T]=CMFH_compress1(model, I_db, T_db, I_te, T_te);

    %% evaluate
    codeTestit=test_hash_X;
    codeTestti=test_hash_T;
    cbDbX = compactbit(codeDbX(:,1:bits));
    cbDbY = compactbit(codeDbY(:,1:bits));
    cbTestit  = compactbit(codeTestit(:,1:bits));
    cbTestti  = compactbit(codeTestti(:,1:bits));
    hamm_I2T  = hammingDist(cbTestit,cbDbY )';
    hamm_T2I  = hammingDist(cbTestti,cbDbX )';

    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

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

