addpath('libsvm');
load('ovarian-cancer_data.mat');
folds=5;                	%no of folds
for i= 1:folds;
    
	train_Features= train_data{i};
	train_Labels= train_gp{i};
	test_Features= test_data{i};
	test_Labels= test_gp{i};
	[M,N] = size(train_Features);
	[M,N] = size(train_Features);
 
	train_Features= mn_var_norm(train_Features);
	test_Features= mn_var_norm(test_Features);
    
% hlevel_data = train_Features;
% test_hlevel_data = test_Features;
% trainLabels = train_Labels;
% testLabels = test_Labels;


% % Defining the parameters
%
% Parameter.ker = 'linear';
% Parameter.p1 = 0.2;
% Parameter.CC = 8;
% Parameter.CR = 1;
% Parameter.p1 = 0.2;
% Parameter.v = 10;
% Parameter.algorithm = 'CD';    
% Parameter.showplots = true;
%
% fprintf('1st parameters defined....\n');
%
% ker=Parameter.ker;
% CC=Parameter.CC;
% CR=Parameter.CR;
% Parameter.autoScale=0;
%
%
% Parameter.showplots=0;
%
% autoScale=Parameter.autoScale;
%
%
% st1 = cputime;

% Classifying the labels as +1 and -1

% [groupIndex, groupString] = grp2idx(train_Labels);
% groupIndex = 1 - (2* (groupIndex-1));
% % scaleData = [];
% 
% % scTraindata= hlevel_data;
% 
% %Separating the positive and negative datapoints and labels
% 
% Xp=train_Features(groupIndex==1,:);
% Lp=train_Labels(groupIndex==1);
% Xn=train_Features(groupIndex==-1,:);
% Ln=train_Labels(groupIndex==-1);
% 
% %combining the positive and negative data samples
% 
% X=[Xp;Xn];
% L=[Lp;Ln];
% 
% 
% 
% fprintf('Positive and negative data separated....\n');
% 
% 
% %Training SVM to find out the weights
% 
% % SVMXp= svmtrain(X,L);
% SVMXp1= fitcsvm(X,L);
% 
% u=0.1;
% eplison=1e-7;
% [rxp,cxp]=size(Xp);
% [rxn,cxn]=size(Xn);
% X=[Xp;Xn];
% 
% 
% %linear kernel
% 
% kfun = @linear_kernel;
% kfunargs ={};
% 
% %distance of positive points from positive hyperplane
% %calculating weights
% 
%  w= SVMXp1.W;
%  bias=SVMXp1.Bias;
%  w=w'*ones(size(X));
%  for y=1:rxp
%   	prod1(y)= w*(Xp(y,:))'+ bias - 1;
%   	ww=w*w'+bias-1;
%   	prod1(y)=prod1(y)/ww;
%   	radiusxp1(y)= prod1(y);
%  end
%  radiusxpmax1=max(radiusxp1);
%  
% 
%  
%  fprintf('weights calculated....\n');
%  fprintf('positve distance calculated from positive hyperplane....\n');
%  
%  %distance of positive points from negative hyperplane
%  
%  w= SVMXp1.W;
%  bias=SVMXp1.Bias;
%  w=w'*ones(size(X));
%  for y=1:rxp
% 	prod1(y)= w*(Xp(y,:))'+ bias + 1;
% 	ww=w*w'+bias+1;
% 	prod1(y)=prod1(y)/ww;
% 	radiusxp2(y)= prod1(y);
%  end
%  radiusxpmax2=max(radiusxp2);
% 
% 
%  fprintf('positive distance calculated from negative hyperplane....\n');
%  
%  %distance of negative points from positive hyperplane
%  
%  w= SVMXp1.W;
%  w=w'*ones(size(X));
%  for y=1:rxn
%  	prod(y)= w*(Xn(y,:))'+ bias - 1;
%  	ww=w*w'+bias-1;
%  	prod(y)=prod(y)/ww;
%  	radiusxn1(y)= prod(y);
%  end
%  radiusxnmax1=max(radiusxn1);
% 
%        	 
%  fprintf('negative distance calculated from positive hyperplane....\n');
%  %distance of negative points from negative hyperplane
%  
%  w= SVMXp1.W;
%  w=w'*ones(size(X));
%  for y=1:rxn
%  	prod(y)= w*(Xn(y,:))'+ bias + 1;
%  	ww=w*w'+bias+1;
%  	prod(y)=prod(y)/ww;
%  	radiusxn2(y)= prod(y);
%  	end
%  radiusxnmax2=max(radiusxn2);
%      	 
%  
%  fprintf('negative distance calculated from negative hyperplane....\n');  
%  
%  clear w ww bias SVMXp cxp cxn;
%  clear Lp Ln L;
%  
%  %upper and lower membership degrees of positive and negative points
%  
%  sp1=exp(-u.*(abs(radiusxp1)./(radiusxpmax1)));
%  sp2=exp(-u.*(abs(radiusxp2)./(radiusxpmax2)));
%  sn1=exp(-u.*(abs(radiusxn1)./(radiusxnmax1)));
%  sn2=exp(-u.*(abs(radiusxn2)./(radiusxnmax2)));
%  
%  fprintf('upper and lower membership degrees calculated....\n');
%  
%  %mean of positive and negative membership degrees
%  
% %  sp=(sp1+sp2)/2;
% %  sn=(sn1+sn2)/2;
% 
%  
% 
% 
% index1=[sp1';sn1'];
% index2=[sp2';sn2'];
% 
% clear sp1 sp2 sn1 sn2 radiusxp1 radiusxp2 radiusxn1 radiusxn2 prod prod1 radiusxpmax1 radiusxpmax2 radiusxnmax1 radiusxnmax2;
% 
% %calculating MU upper
% 
% for k=1:N
% 	tmp1=index1';
% 	nume1= tmp1*train_Features(:,k);
% 	deno1= sum(tmp1);
% 	Mu_upper(k,:)= nume1/deno1;
% end
% 
% fprintf('MU UPPER Calculated....\n');
% clear temp1 nume1 deno1 k;
% 
% 
% %calculating MU Lower
% 
% for k=1:N
% 	tmp2= index2';
% 	nume2= tmp2*train_Features(:,k);
% 	deno2= sum(tmp2);
% 	Mu_lower(k,:)= nume2/deno2;
% end
% 
% fprintf('MU LOWER Calculated....\n');
% clear tmp2 nume2 deno2 k;
% 
% 
% % Mu_average= (Mu_upper+Mu_lower)./2;
% 
% %Upper covarience matrix
% 
% 	Cov_mat_upper=0;
% 	for ii=1:M
%     	dvect=train_Features(ii,:)-Mu_upper';
%     	Cov_mat_upper = Cov_mat_upper + (index1(ii).^2)*(dvect')*dvect;
% 	end
% 
% 	fprintf('Upper Covarience matrix calculated...\n');
% 	clear index1 dvect ii;
%     
% %lower covarience matrix
% 
% 	Cov_mat_lower=0;
% 	for ii=1:M
%     	dvect=train_Features(ii,:)-Mu_lower';
%     	Cov_mat_lower= Cov_mat_lower + (index2(ii).^2)*(dvect')*dvect;
% 	end
% 
% 	clear index2 dvect ii;
%     
% 	fprintf('Lower covarience matrix calculated....\n');
% 
% %average covarience matrix
% 
% Cov_mat_avg1= (Cov_mat_upper+Cov_mat_lower)./2;
% 
% clear Cov_mat_upper Cov_mat_lower;
% % Mu_average= (Mu_upper+Mu_lower)./2;
% % index_mean= (index1+index2)/2;
% 
% 
% % for ik=1:N
% % 	for ii=1:M
% %     	dvect=train_Features(ii,:)-Mu_average(ik,:);
% %     	Cov_mat_avg2= index_mean(ii)*(dvect')*dvect;
% % 	end
% % end
% 
% Cov_mat= Cov_mat_avg1/M;
% 
% clear Cov_mat_avg1;

%
% min= min(min(Cov_mat));
% max=max(max(Cov_mat));
%	 
%	% zero _one_normalization
%     Cov_mat = (Cov_mat - min*(ones(size(Cov_mat))))./(max - min);
%	 
% 	fprintf('Covarience matrix normalised....\n');
  % PCA
%  [U,S,V] = svd(Cov_mat);
% no_features= 17;
% V_new= V(:,(1:no_features));
% new_train_data= train_Features*V_new;
% new_test_data= test_Features*V_new;

%Feature selection using PCA

 [U,S,V] = svd(train_Features);
% [COEFF,score,explained,latent,tsquare] = pca(Cov_mat);
 no_features=500;
 V_new= V(:,(1:no_features));
 new_train_data= train_Features*V_new;
 new_test_data= test_Features*V_new;
 
 clear U S V;
 clear Cov_mat;
 
 fprintf('Feature selection done using pca...\n');
 
%new data combined with labels
%new_train_data= [new_train_data,train_Labels];
%new_test_data= [new_test_data,test_Labels];

hlevel_data = new_train_data;
test_hlevel_data = new_test_data;
trainLabels = train_Labels;
testLabels = test_Labels;

fprintf('SVM started...\n');


bestcv = 0;
    bestc=0;
    %%%%%%%%tuning of parameters%%%%%%%%%
    for log2c = 1:15
        cmd = ['-t 0 -v 5 -c ', num2str(2^log2c), ' -q'];
        cv = svmtrain(trainLabels, hlevel_data, cmd);
        if (cv >= bestcv)
            bestcv = cv; 
            bestc = log2c;
        end
        fprintf('%g %g (best c = %g, rate=%g)\n', log2c, cv, bestc, bestcv);
    end
    
    cur_bestc = bestc;
    
   %Finer grid search for finding best parameters
    for ii = 1:7
        log2c = cur_bestc - 1 + 0.25*ii;
        cmd = ['-t 0 -v 5 -c ', num2str(2^log2c), ' -q'];
        cv = svmtrain(trainLabels, hlevel_data, cmd);
        if (cv >= bestcv)
            bestcv = cv; bestc = 2^log2c;
        end
        fprintf('%g %g (best c = %g, rate=%g)\n', log2c, cv, bestc, bestcv);
    end
    
    cmd = ['-t 0 -c ', num2str(bestc),' -b 1'];
    
%    Final model made ready
    clsfr_model = svmtrain(trainLabels,hlevel_data, cmd);
    
    [pred_label_lin, acc1, prob_est1] = svmpredict(testLabels, test_hlevel_data, clsfr_model); %%%'-b 1'
%     
    [~,~,T,auc_rf(i),optrocpt] = perfcurve(testLabels,pred_label_lin(:,1),2);
     acc_lin(i) = acc1(1);

   
    bestcv = 0;
    bestg=0;
    % with RBF kernel
    % Finding the best parameters through grid search
    for log2c = 1:15
        for log2g = -15:-1
            cmd = ['-t 2 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' -q'];
            cv = svmtrain(trainLabels, hlevel_data, cmd);
            if (cv >= bestcv)
                bestcv = cv; 
                bestc = log2c; 
                bestg = log2g;
            end
            fprintf('%g %g %g (best c = %g, best g = %g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
        end
    end
    
    cur_bestc = bestc; cur_bestg = bestg;
    
    % Finer grid search for finding best parameters
    for ii = 1:7
        log2c = cur_bestc - 1 + 0.25*ii;
        for k = 1:7
            log2g = cur_bestg - 1 + 0.25*k;
            cmd = ['-t 2 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' -q'];
            cv = svmtrain(trainLabels, hlevel_data, cmd);
            if (cv >= bestcv)
                bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
            end
            fprintf('%g %g %g (best c = %g, best g = %g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
        end
    end
    
    cmd = ['-t 2 -c ', num2str(bestc), ' -g ', num2str(bestg),' -b 1'];
    
    % Final model made ready
    clsfr_model = svmtrain(trainLabels, hlevel_data, cmd);
    
    [pred_label_rbf, acc2, prob_est2] = svmpredict(testLabels, test_hlevel_data, clsfr_model);%%, '-b 1');
    
%       [~,~,T,auc_rbf(j)] = perfcurve(testLabels,prob_est2,2);
    acc_rbf(i) = acc2(1);

% random forest
    
    B = TreeBagger(1000,hlevel_data,trainLabels);
 	[Yfit,scores,stdevs] = predict(B,test_hlevel_data);
    Yfit=cell2mat(Yfit);
    Yfit=str2num(Yfit);
    acc_Rf(i) = mean(testLabels(:) == Yfit(:));

 %softmax
 
 [acc(i),prob,pred] = softmaxExercise(hlevel_data,trainLabels,test_hlevel_data,testLabels);
  [~,~,T1,auc(i)] = perfcurve(testLabels,prob(1,:),1);
 pred_softmax=pred';

 

%  acc(i)= ftsvmclass(ftsvm_struct,test_hlevel_data,testLabels);
clear max;
clear min;

% s = diag(S); % Vector of singular values

% % Plot cumulative variance explained by first k modes
%
% normsqS = sum(s.^2);
% figure(3)
% clf
% subplot(2,2,1)
% plot(cumsum(s.^2)/normsqS,'x')  % Cumulative fraction of variance explained
% xlabel('Mode k')
% ylabel('Cumulative Variance fraction explained')
% ylim([0.9 1])

% C = S(1:20,1:20)*V(:,1:20)';   % first 3 coefficients for each point, same as U(:,1:3)'*A;
%
% sigma = diag(S);        	% singular values
%
% rho = norm(sigma)^2;	 
% q2 = norm(sigma(1:2))^2/rho;  % part of variation captured by first 2 components
% figure(5);
% plot(C(1,:),C(2,:),'.')
% xlabel('PC1'); ylabel('PC2')

end
