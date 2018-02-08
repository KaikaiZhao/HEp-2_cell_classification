load('features_vgg_cell_finetune1234.mat')

addpath(genpath('/home/lib601/libsvm'));

[newX,T,meanValue] = mypca(feature,0.995);
%[newX6,T,meanValue] = mypca(v36,0.995);
%v = [v36,v37];
% [newX,T,meanValue] = mypca(v,0.995);

[c,g]=meshgrid(-10:10:10,-10:10:10);
[m,n]=size(c);
cg=zeros(m,n);
eps=10^(-4);
v = 5;
bestc = 1;
bestg = 0.1;
bestacc = 0;
trnum = 200;
a=randperm(229);
newX2=newX(a,:);
train_x = newX2(1:trnum,:);
test_x =  newX2(1+trnum : 229, :);
load label.mat
label2=label;
train_y = label2(1:trnum);
test_y =  label2(1+trnum : 229);

for i = 1:m
    for j = 1:n
         cmd = ['-v ',num2str(v),' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j)),'-t 0'];
         disp('start train...');
         cg(i,j) = svmtrain(train_y,train_x,cmd);
         disp('finish train...');
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end        
        if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j) 
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end               
        disp(j);
    end
end

cmd = [' -t 0',' -c ',num2str(bestc),' -g ',num2str(bestg)];
% ����/ѵ��SVMģ��
model = libsvmtrain(train_y,train_x,cmd);

[predict_label_1,accuracy_1,b_1] = libsvmpredict(train_y,train_x,model,'p');
[predict_label_2,accuracy_2,p_2] = libsvmpredict(test_y,test_x,model,'p');
result_1 = [train_y predict_label_1];
result_2 = [test_y  predict_label_2];
