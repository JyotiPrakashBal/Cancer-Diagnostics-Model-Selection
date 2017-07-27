% sdata=load('ionosphere.csv');
models=cat(2,pred_label_lin,pred_label_rbf,pred_softmax,Yfit);
sdata= cat(2,test_Features,test_Labels);
nbin=15;
total_models=4;
total_fea = size(sdata,2)-1;
samples = size(sdata,1); 
H_mm= zeros(total_models);                  %joint entropy of models                                                    
Red_mm = zeros(total_models);               %Information between the models                                                   
H_FV = zeros(total_models,1);               %entropy of individual models
J_ent = cell(total_models,1);               %joint entropy of model output and target output
H_FC = zeros(total_models,1);               %entropy between model output and target output
MI_fc =zeros(total_models,1);               %Information between the model output and target output
cl(1) = sdata(1,total_fea+1); 
k = 2;

for i =2:samples
    if isempty(find(sdata(i,total_fea+1)==cl))
        cl(k) = sdata(i,total_fea+1); 
        k=k+1;    
    end                 % reading class lable
end


cl = sort(cl);
if cl(1) == 0
    cl = cl+1;
    sdata(:,total_fea+1)=sdata(:,total_fea+1)+1; 
end
nc = max(cl); 


% Calculation of Joint entropy and frequencys for models
for i = 1:total_models, 
    J_ent{i}=zeros(nc,nbin); 
end
 range = zeros(total_models,nbin);
for i = 1:total_models-1
    range(i,:) = min(models(:,i)):(max(models(:,i))-min(models(:,i)))/(nbin-1):max(models(:,i));
    for j = 1:samples        
        bi = min(find(models(j,i)<range(i,:)));
        if(isempty(bi)), 
            bi = nbin; 
        else
            bi = bi-1;        
        end
        J_ent{i}(models(j,i),bi) = J_ent{i}(models(j,i),bi)+1;
    end
    FV(i,:) = sum(J_ent{i});
end

% Calculation of Entropy of class H_C
c(nc) = 0;
for i=1:samples
    k = find(sdata(i,total_fea+1)==cl);
    c(k)= c(k)+1;
end

c = c./samples;
H_C = -sum(sum(c.*log2(c)));

% Calculation of Entropy of individual models
for i =1 : total_models-1
    fv = FV(i,find(FV(i,:)));
    fv= fv/sum(sum(fv));
    H_FV(i) = -sum(sum(fv.*log2(fv)));    
end

% Calculation of Entropy between all model pairs H_mm
Edge = cell(2,1);
for i = 1:total_models
    Edge{1}=range(i,:);
    for j = 1:total_models
        Edge{2}=range(j,:);      
        [x y] = hist3([models(:,i),models(:,j)],'edges',Edge);
        x= x/sum(sum(x));
        x = x(find(x));
        H_mm(i,j) = -sum(sum(x.*log2(x)));
    end    
end 

% Caclulating mutual info between fv and class
for i = 2 : total_models
    a = J_ent{i};
    a= a/sum(sum(a));
    z = a(find(a));
    H_FC(i) = -sum(sum(z.*log2(z)));
end

% Calculation of Mutual Information
for i =1 :total_models
    MI_fc(i) = H_FV(i)+H_C-H_FC(i);    
    for j = 1:total_models
        MI_ff(i,j) = H_FV(i)+ H_FV(j)-H_mm(i,j);
    end
%     disp(strcat('MI calculation 2nd last stage for Feature_',int2str(i)));
end

num_models=4;            %number of models to be selected     

nbin = 15;               % number of bin for frequency calcuation
fvdim = total_models;    % total number of models

K = fvdim;              % |S| = k ;
S = zeros(K,1);
F = 1:fvdim;
[a S(1)]= max(MI_fc);
 F = F(find(F~=S(1)));
 %count = 1;
for k = 2 : num_models-1
    clear mid
    %mid = parfor_part_MIind(MI_ff,Flag,F,k,S);
    mid = zeros(size(F,2),1);
    for i =1 : total_models-1   
            summ=0;
            %for j = 1 : k-1            
            %    summ = summ + MI_ff(F(i),S(j));            
            %end
			summ = sum(MI_ff(F(i),S(1:k-1)));
            mid(i) = MI_fc(F(i))- summ/(k-1);
        
    end
    [a b] = max(mid);
    S(k) = F(b);
%     F = F(find(F~=S(k)));
    %count = count + 1; 
end

model_ind = S(1:num_models);              %ranking of models
