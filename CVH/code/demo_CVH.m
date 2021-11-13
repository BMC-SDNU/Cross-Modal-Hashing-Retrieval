
function [] = demo_CVH(bits, dataname)
    %clc;
    %clear;
    % vl_setup;
    k_near = 5;
	bits = str2num(bits);

    addpath('../../Data');
    conf.randSeed = 1;
    randn('state',conf.randSeed) ;
    rand('state',conf.randSeed) ;

    if dataname == 'flickr'
        load('mir_cnn.mat');
    elseif dataname == 'nuswide'
        load('nus_cnn.mat');
    elseif dataname == 'coco'
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end

	feaTrain_text = T_tr;
	clear T_tr;
	feaTrain_vis = I_tr;
	clear I_tr;
	feaTrain_class = L_tr;
	clear L_tr;

	feaTest_text = T_te;
	clear T_te;
	feaTest_vis = I_te;
	clear I_te;
	feaTest_class =L_te;

	feaDb_text = T_db;
	clear T_db;
	feaDb_vis = I_db;
	clear I_db;
	feaDb_class =L_db;

	CVHparam.lambda=1e-4;

	[model, drop]=CVH_learn(feaTrain_vis , feaTrain_text,CVHparam, bits);

	codeDb_T = CVH_compress1(feaDb_text, model);
	codeDb_I = CVH_compress(feaDb_vis, model);

	codeTest_T = CVH_compress1(feaTest_text, model);
	codeTest_I = CVH_compress(feaTest_vis, model);

	%% evaluation
	codeDb_T = codeDb_T';
	codeDb_I = codeDb_I';

	codeTest_T = codeTest_T';
	codeTest_I = codeTest_I';

	cbDb_T = compactbit(codeDb_T(:,1:bits));
	cbDb_I = compactbit(codeDb_I(:,1:bits));
	
	cbTest_T = compactbit(codeTest_T(:,1:bits));
	cbTest_I = compactbit(codeTest_I(:,1:bits));

	hamm_T2I = hammingDist(cbTest_T, cbDb_I)';
	MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

	hamm_I2T = hammingDist(cbTest_I, cbDb_T)';
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
