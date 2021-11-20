function [] = demo_CCQ(bits, dataname)
    warning off;
    addpath('../../Data');
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
    bookbits = 8;
    k = 2 ^ bookbits;
    knn = 100;

    nbits = [16 32 64 128];

    %top_num = 1;
    lambda = 1;
    run = 5;
    %tot_loop = length(lambdas) * length(ds);


    I_tr = zscore(I_tr);
    I_te = zscore(I_te);
    T_tr = zscore(T_tr);
    T_te = zscore(T_te);
    n0 = size(I_tr, 1);
    %d = 10; %10

    Xbase = I_db';
    Ybase = T_db';
    Lbase = L_db;
    Xquery = I_te';
    Yquery = T_te';
    Lquery = L_te;
    Xtrain = I_tr';
    Ytrain = T_tr';

	d = bits;
    % fprintf('bits: %d\n', bits);
    m = bits / bookbits;

    idx = 1;
    
    % Training
    model = CCQ(Xtrain, Ytrain, n0, d, m, k, lambda);

    %fprintf('quantize database & query...');
    XbaseQ = ccq_encode(model, Xbase, []);
    YbaseQ = ccq_encode(model, [], Ybase);
    XYbaseQ = ccq_encode(model, Xbase, Ybase);
    [XqueryQ, XqueryR] = ccq_encode(model, Xquery, []);
    [YqueryQ, YqueryR] = ccq_encode(model, [], Yquery);
    XbaseL2 = double(sum(ccq_decode(model, XbaseQ, []).^2, 1));
    YbaseL2 = double(sum(ccq_decode(model, [], YbaseQ).^2, 1));
    XYbaseL2 = double(sum(ccq_decode(model, XYbaseQ, []).^2, 1));
    %fprintf('done!\n');

    hamm_T2I = ccq_linscan(YqueryR, XbaseQ, XbaseL2, model.Cx, knn)';
    MAP_T2I = perf_metric4Label(Lbase, Lquery, -hamm_T2I);

    hamm_I2T = ccq_linscan(XqueryR, YbaseQ, YbaseL2, model.Cy, knn)';
    MAP_I2T = perf_metric4Label(Lbase, Lquery, -hamm_I2T);

    
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

